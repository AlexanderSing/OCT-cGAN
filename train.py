import pytorch_lightning as pl
import torch.utils.data
from pytorch_lightning.loggers import TensorBoardLogger

import config
from ImageLoader import ImageLoader
from Model import Pix2PixModel

if __name__ == '__main__':
    # Set seeds
    pl.seed_everything(config.SEED)

    if config.VALSET_PATH:
        train_dataset = ImageLoader(root_path=config.DATASETS_PATH,
                                    image_size=config.INPUT_SIZE,
                                    transform=config.TRAIN_TRANSFORMS,
                                    training=True,
                                    crops=config.CROPS
                                    )
        val_dataset = ImageLoader(root_path=config.VALSET_PATH,
                                  image_size=config.INPUT_SIZE,
                                  transform=config.DEPLOYMENT_TRANSFORMS,
                                  training=True,
                                  crops=config.CROPS
                                  )
        train_set_size = len(train_dataset)
        val_set_size = len(val_dataset)
    else:
        # Load datasets and create data loaders afterwards
        dataset = ImageLoader(root_path=config.DATASETS_PATH,
                              image_size=config.INPUT_SIZE,
                              transform=config.TRAIN_TRANSFORMS,
                              training=True,
                              crops=config.CROPS
                              )

        train_set_size = int(
            min(len(dataset) - config.FEATURE_DIMENSIONALITY_FID, len(dataset) * config.TRAIN_FRACTION))
        val_set_size = len(dataset) - train_set_size

        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_set_size, val_set_size])

    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=config.BATCH_SIZE,
                                                    shuffle=True,
                                                    pin_memory=True)

    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  pin_memory=True)

    model = Pix2PixModel(input_size=config.INPUT_SIZE,
                         batch_size=config.BATCH_SIZE,
                         generator_filters_base=config.GENERATOR_FILTER_BASE,
                         generator_depth=config.GENERATOR_DEPTH,
                         generator_layers_per_level=config.GENERATOR_LAYERS_PER_LEVEL,
                         generator_use_dropout=config.USE_DROPOUT,
                         discriminator_filters_base=config.DISCRIMINATOR_FILTER_BASE,
                         discriminator_n_layers=config.DISCRIMINATOR_N_LAYERS,
                         discriminator_convs_per_layer=config.DISCRIMINATOR_CONVS_PER_LAYER,
                         discriminator_trains_per_image=config.DISCRIMINATOR_TRAINS_PER_IMAGE,
                         discriminator_use_sigmoid=config.USE_SIGMOID,
                         gan_criterion=config.GAN_LOSS,
                         l_criterion=config.L_LOSS,
                         optimizer=config.OPTIMIZER,
                         learning_rate_d=config.LEARNING_RATE_D,
                         betas_d=config.BETAS_D,
                         learning_rate_g=config.LEARNING_RATE_G,
                         betas_g=config.BETAS_G,
                         lambda_l=config.LAMBDA_L,
                         pool_size=config.POOL_SIZE,
                         feature_dimensionality_fid=config.FEATURE_DIMENSIONALITY_FID,
                         train_set_size=train_set_size,
                         val_set_size=val_set_size)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=5, monitor='SSIM', mode='max')
    trainer = pl.Trainer(gpus=config.GPUS, deterministic=True, max_epochs=config.MAX_EPOCHS,
                         logger=TensorBoardLogger("lightning_logs", name="", log_graph=True),
                         callbacks=[checkpoint_callback])

    trainer.fit(model, train_dataloader=train_data_loader, val_dataloaders=val_data_loader)
