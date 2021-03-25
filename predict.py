import os
from argparse import ArgumentParser

import cv2
import numpy as np
import time
import pytorch_lightning as pl
import torch.utils.data

import config
from ImageLoader import ImageLoader
from Model import Pix2PixModel

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_path')
    parser.add_argument('--data_path')
    parser.add_argument('--output_path')
    parser.add_argument('--phase', action='store_true')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--output_format', default='tiff')

    args = parser.parse_args()

    # Set seeds
    pl.seed_everything(config.SEED)

    # Load datasets and create data loaders afterwards
    dataset = ImageLoader(root_path=args.data_path,
                          image_size=config.INPUT_SIZE,
                          transform=config.DEPLOYMENT_TRANSFORMS,
                          training=False)

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              pin_memory=True)

    model = Pix2PixModel.load_from_checkpoint(args.model_path).cuda()
    model.freeze()

    created_dir = False
    k = 0
    while not created_dir:
        try:
            os.mkdir(args.output_path + str(k))
            created_dir = True
        except OSError as error:
            k += 1

    for i, input_image in enumerate(data_loader):
        result = model.forward(input_image.cuda())
        for j in result:
            img = j.detach().cpu().numpy()
            img = np.moveaxis(img, 0, -1)
            img = img * 255
            img = img.astype(np.uint8)
            cv2.imwrite(args.output_path + str(k) + '/' + str(i) + "." + args.output_format, img)

