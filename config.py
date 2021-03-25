from typing import Tuple

import torch
from torchvision import transforms

SEED = 999

DATASETS_PATH = "All_Synthetic"
INPUT_FOLDER_NAME = "Gaped"
REFERENCE_FOLDER_NAME = "Full"

VALSET_PATH = ""

AMP_FOLDER_NAME = "amp"
PHASE_FOLDER_NAME = "phase"

GPUS = 1

TRAIN_FRACTION = 0.95

BATCH_SIZE: int = 1
INPUT_SIZE: Tuple[int, int, int] = (2, 1024, 301)

GENERATOR_FILTER_BASE = 64
GENERATOR_DEPTH = 7
GENERATOR_LAYERS_PER_LEVEL = 1

DISCRIMINATOR_FILTER_BASE = 64
DISCRIMINATOR_N_LAYERS = 3
DISCRIMINATOR_CONVS_PER_LAYER = 1
DISCRIMINATOR_TRAINS_PER_IMAGE = 3

OPTIMIZER = torch.optim.AdamW
LEARNING_RATE_D = 0.0002
BETAS_D = (0.5, 0.999)

LEARNING_RATE_G = 0.0002
BETAS_G = (0.5, 0.999)

GAN_LOSS = torch.nn.BCELoss()
L_LOSS = torch.nn.L1Loss()

LAMBDA_L = 100.0

USE_DROPOUT = True
USE_SIGMOID = True

POOL_SIZE = 50

FEATURE_DIMENSIONALITY_FID = 192

MAX_EPOCHS = 100

CROPS = ((128, 384), (0, 256))

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])

DEPLOYMENT_TRANSFORMS = transforms.Compose([
    transforms.ToTensor()
])
