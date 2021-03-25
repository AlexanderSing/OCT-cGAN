from typing import Tuple, List, Union, Type

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.utils.data
from piq import ssim, psnr

from Pix2PixPatchDiscriminator import Discriminator
from Pix2PixUNETGenerator import Generator
from ImageLoader import ImageLoader
from utils import ImagePool, get_label_tensor, initialize_weights, tensor2im, fid


class Pix2PixModel(pl.LightningModule):
    def __init__(self,
                 input_size: tuple = (1, 384, 384),
                 batch_size: int = 1,
                 generator_filters_base: int = 64,
                 generator_depth: int = 7,
                 generator_layers_per_level: int = 1,
                 generator_use_dropout: bool = False,
                 discriminator_filters_base: int = 64,
                 discriminator_n_layers: int = 3,
                 discriminator_convs_per_layer: int = 1,
                 discriminator_trains_per_image: int = 3,
                 discriminator_use_sigmoid: bool = True,
                 gan_criterion: nn.Module = nn.BCELoss(),
                 l_criterion: nn.Module = nn.L1Loss(),
                 pan_criterion: nn.Module = nn.L1Loss(),
                 optimizer: Type[torch.optim.Optimizer] = torch.optim.AdamW,
                 learning_rate_d: float = 0.0002,
                 betas_d: tuple = (0.5, 0.999),
                 learning_rate_g: float = 0.0002,
                 betas_g: tuple = (0.5, 0.999),
                 lambda_l: float = 100.0,
                 lambdas_pan: Tuple[float] = (5.0, 1.5, 1.5, 1.5, 1.0),
                 pan_margin: int = 50,
                 pool_size: int = 10,
                 feature_dimensionality_fid: int = 192,
                 train_set_size: int = 1000,
                 val_set_size: int = 0):

        super().__init__()

        self.save_hyperparameters()
        self.generator = Generator(input_channels=input_size[0],
                                   output_channels=1,
                                   depth=generator_depth,
                                   generator_filters_base=generator_filters_base,
                                   use_dropout=generator_use_dropout,
                                   layers_per_level=generator_layers_per_level)
        self.discriminator = Discriminator(input_channels=2,
                                           discriminator_filters_base=discriminator_filters_base,
                                           n_layers=discriminator_n_layers,
                                           convs_per_layer=discriminator_convs_per_layer,
                                           use_sigmoid=discriminator_use_sigmoid)

        self.image_pool: Union[ImagePool, None] = None

        initialize_weights(self.generator)
        initialize_weights(self.discriminator)
        self.example_input_array = torch.zeros([1, *input_size])
        self.loss_PAN = 0
        self.reference_images = torch.zeros([self.hparams.val_set_size, 3, self.hparams.input_size[1], self.hparams.input_size[2]])
        self.forged_images = torch.zeros([self.hparams.val_set_size, 3, self.hparams.input_size[1], self.hparams.input_size[2]])

    def on_train_start(self) -> None:
        self.image_pool = ImagePool(pool_size=self.hparams.pool_size,
                                    input_size=(self.hparams.batch_size,
                                                2,
                                                self.hparams.input_size[1],
                                                self.hparams.input_size[2]),
                                    device=self.device)

    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        return self.generator(input_image)

    def generator_loss(self, input_images: torch.Tensor, reference_images: torch.Tensor) -> torch.Tensor:
        forged_images = self(input_images)

        forged_with_input = torch.cat((input_images[:, 0:1], forged_images), 1)
        output = self.discriminator(forged_with_input)

        label = get_label_tensor(output, 1.0, self.device)

        loss_G_GAN = self.hparams.gan_criterion(output, label)

        loss_G_L = self.hparams.l_criterion(forged_images, reference_images[:, 0:1]) * self.hparams.lambda_l

        self.log('g_GAN_loss', loss_G_GAN.detach(), on_step=True, on_epoch=False)
        self.log('g_L_loss', loss_G_L.detach(), on_step=True, on_epoch=False)

        return loss_G_GAN + loss_G_L + self.loss_PAN

    def discriminator_loss(self, forged_with_input: torch.Tensor, reference_with_input: torch.Tensor) -> torch.Tensor:
        output = self.discriminator(reference_with_input)
        label = get_label_tensor(output, 1.0, self.device)

        real_intermediate_outputs = self.discriminator.get_intermediate_output()

        loss_D_real = self.hparams.gan_criterion(output, label)

        output = self.discriminator(forged_with_input)
        label = get_label_tensor(output, 0.0, self.device)

        forged_intermediate_outputs = self.discriminator.get_intermediate_output()

        loss_D_forged = self.hparams.gan_criterion(output, label)

        loss_D_PAN = torch.zeros(1, device=self.device)

        for (forged_i, real_i, lam) in zip(forged_intermediate_outputs, real_intermediate_outputs, self.hparams.lambdas_pan):
            loss_D_PAN += self.hparams.pan_criterion(forged_i, real_i) * lam

        self.loss_PAN = loss_D_PAN.detach()

        loss_D_PAN = max(torch.zeros(1, device=self.device), torch.tensor(self.hparams.pan_margin, device=self.device) - loss_D_PAN)

        self.log('d_PAN_loss', loss_D_PAN.detach(), on_step=True, on_epoch=False)
        self.log('d_real_loss', loss_D_real.detach(), on_step=True, on_epoch=False)
        self.log('d_forged_loss', loss_D_forged.detach(), on_step=True, on_epoch=False)

        return (loss_D_real + loss_D_forged) * 0.5 + loss_D_PAN

    def training_step(self, batch, batch_idx, optimizer_idx) -> torch.Tensor:
        input_images, reference_images = batch

        result = None

        if optimizer_idx == 0:
            result = torch.zeros(size=(self.hparams.discriminator_trains_per_image, 1))

            forged_images = self(input_images)
            forged_with_input = torch.cat((input_images[:, 0:1], forged_images), 1)
            reference_with_input = torch.cat((input_images[:, 0:1], reference_images[:, 0:1]), 1)

            if self.hparams.pool_size > 0:
                self.image_pool.add_image(forged_with_input.detach(), False)
                self.image_pool.add_image(reference_with_input.detach(), True)

            result[0] = self.discriminator_step(forged_with_input=forged_with_input.detach(),
                                                reference_with_input=reference_with_input)

            for i in range(1, self.hparams.discriminator_trains_per_image):
                result[i] = self.discriminator_step(forged_with_input=self.image_pool.get_image(True),
                                                    reference_with_input=self.image_pool.get_image(False))

            result = torch.mean(result)

        if optimizer_idx == 1:
            result = self.generator_step(input_images=input_images, reference_images=reference_images[:, 0:1])

        return result

    def generator_step(self, input_images: torch.Tensor, reference_images: torch.Tensor) -> torch.Tensor:
        g_loss = self.generator_loss(input_images, reference_images)

        self.log('g_loss', g_loss.detach(), on_step=True, on_epoch=False)
        return g_loss

    def discriminator_step(self, forged_with_input: torch.Tensor, reference_with_input: torch.Tensor) -> torch.Tensor:
        d_loss = self.discriminator_loss(forged_with_input=forged_with_input, reference_with_input=reference_with_input)

        self.log('d_loss', d_loss.detach(), on_step=True, on_epoch=False)
        return d_loss

    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        opt_g = self.hparams.optimizer(self.generator.parameters(),
                                       lr=self.hparams.learning_rate_g,
                                       betas=self.hparams.betas_g)
        opt_d = self.hparams.optimizer(self.discriminator.parameters(),
                                       lr=self.hparams.learning_rate_d,
                                       betas=self.hparams.betas_d)
        return opt_d, opt_g

    def on_validation_epoch_start(self) -> None:
        if isinstance(self.val_dataloader().dataset, ImageLoader):
            self.val_dataloader().dataset.val = True
        else:
            self.val_dataloader().dataset.dataset.val = True

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        input_image, reference_image = batch
        forged_image = self.generator(input_image)
        forged_image = tensor2im(forged_image)
        input_image = tensor2im(input_image)
        reference_image = tensor2im(reference_image)
        if batch_idx % 50 == 0:
            tensorboard = self.logger.experiment
            tensorboard.add_image("Forged", forged_image.cpu().detach())
            tensorboard.add_image("Input", input_image.cpu().detach())
            tensorboard.add_image("Reference", reference_image.cpu().detach())
        self.forged_images[batch_idx] = forged_image
        self.reference_images[batch_idx] = reference_image

    def validation_epoch_end(self, outputs: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        if isinstance(self.val_dataloader().dataset, ImageLoader):
            self.val_dataloader().dataset.val = False
        else:
            self.val_dataloader().dataset.dataset.val = False

        fid_score = fid(self.forged_images, self.reference_images, self.hparams.feature_dimensionality_fid, self.device)
        ssim_score = ssim(self.forged_images, self.reference_images, data_range=255)
        psnr_score = psnr(self.forged_images, self.reference_images, data_range=255)

        self.log('FID_score', fid_score, on_step=False, on_epoch=True)
        self.log('SSIM', ssim_score, on_step=False, on_epoch=True)
        self.log('PSNR', psnr_score, on_step=False, on_epoch=True)
