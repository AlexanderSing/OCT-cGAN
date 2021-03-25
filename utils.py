import os
import random
from typing import Tuple

import torch
import torch.nn as nn
import numpy as np
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3


def tensor2im(input_image: torch.Tensor, imtype: torch.Type = torch.uint8):
    image_tensor = input_image.data[0]

    image_tensor = image_tensor[0].expand(3, -1, -1)  # grayscale to RGB
    image_tensor = image_tensor.clamp(min=0, max=1)
    image_tensor = image_tensor * 255.0  # post-processing: scaling
    return image_tensor.type(imtype)


def fid(forged_images: torch.Tensor, reference_images: torch.Tensor, feature_dimensionality: int, device: torch.device):
    inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[feature_dimensionality]]).to(device)
    inception.eval()

    forged_pred = np.empty((forged_images.shape[0], feature_dimensionality))
    reference_pred = np.empty((reference_images.shape[0], feature_dimensionality))

    for i in range(forged_images.shape[0]):
        forged_image = forged_images[i:i + 1]
        forged_image = forged_image.to(device=device, dtype=torch.float)
        forged_image /= 255
        reference_image = reference_images[i:i + 1]
        reference_image = reference_image.to(device=device, dtype=torch.float)
        reference_image /= 255

        with torch.no_grad():
            pred = inception(forged_image)[0]

        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = nn.functional.adaptive_avg_pool2d(pred, output_size=(1, 1))

        forged_pred[i:i + 1] = pred.cpu().data.numpy().reshape(1, -1)

        with torch.no_grad():
            pred = inception(reference_image)[0]

        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = nn.functional.adaptive_avg_pool2d(pred, output_size=(1, 1))

        reference_pred[i:i + 1] = pred.cpu().data.numpy().reshape(1, -1)

    mu_forged = np.mean(forged_pred, axis=0)
    sigma_forged = np.cov(forged_pred, rowvar=False)

    mu_reference = np.mean(reference_pred, axis=0)
    sigma_reference = np.cov(reference_pred, rowvar=False)

    return calculate_frechet_distance(mu1=mu_forged, sigma1=sigma_forged, mu2=mu_reference, sigma2=sigma_reference)


def set_requires_grad(network: nn.Module, set_grad: bool) -> None:
    for param in network.parameters():
        param.requires_grad = set_grad


def get_label_tensor(pred: torch.Tensor, value: float, device: torch.device) -> torch.Tensor:
    tensor = torch.tensor(value, dtype=torch.float, device=device)
    return tensor.expand_as(pred)


def initialize_weights(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        nn.init.normal_(m.weight.data, 0.0, 0.02)

        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def get_loss_function(name: str) -> nn.Module:
    if name == 'MSE':
        return nn.MSELoss()
    elif name == 'L1':
        return nn.L1Loss()


def split_path(path: str) -> list:
    all_parts = []
    while True:
        parts = os.path.split(path)
        if parts[0] == path:
            all_parts.insert(0, parts[0])
            break
        elif parts[1] == path:
            all_parts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            all_parts.insert(0, parts[1])

    return all_parts


class ImagePool:

    def __init__(self, pool_size: int, input_size: Tuple[int, int, int, int], device: torch.device) -> None:
        self.pool_size = pool_size
        self.real_images = torch.zeros(size=(pool_size, *input_size), device=device)
        self.forged_images = torch.zeros(size=(pool_size, *input_size), device=device)
        self.number_current_real_images = 0
        self.number_current_forged_images = 0

    def add_image(self, image: torch.Tensor, label: bool) -> None:
        if label:
            if self.number_current_real_images < self.pool_size - 1:
                self.real_images[self.number_current_real_images] = image
                self.number_current_real_images += 1
            else:
                if random.random() > 0.5:
                    replace_index = random.randint(0, self.pool_size - 1)
                    self.real_images[replace_index] = image
        else:
            if self.number_current_forged_images < self.pool_size - 1:
                self.forged_images[self.number_current_forged_images] = image
                self.number_current_forged_images += 1
            else:
                if random.random() > 0.5:
                    replace_index = random.randint(0, self.pool_size - 1)
                    self.forged_images[replace_index] = image

    def get_image(self, label: bool) -> torch.Tensor:
        if label:
            return_index = random.randint(0, self.number_current_real_images)
            return self.real_images[return_index]
        else:
            return_index = random.randint(0, self.number_current_forged_images)
            return self.forged_images[return_index]
