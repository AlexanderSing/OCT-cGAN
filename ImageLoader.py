import os
from typing import List, Union, Tuple

import cv2
import numpy as np
import torch
from torchvision.datasets import DatasetFolder
from torchvision.transforms import ToPILImage, Compose, ToTensor

import config


def make_dataset(root: str, files: List) -> List[str]:
    images = []

    for image in sorted(files):
        path = os.path.join(root, image)
        images.append(path)

    return images


class ImageLoader(DatasetFolder):
    def __init__(self,
                 root_path: str,
                 image_size: Tuple[int, int, int],
                 transform: Compose = None,
                 training: bool = True,
                 crops: Tuple[Tuple[int, int], Tuple[int, int]] = ((0, 384), (0, 384))):

        super(DatasetFolder, self).__init__(root_path, transform=transform, target_transform=None)

        if image_size[0] == 1:
            phase = False
        else:
            phase = True

        folders = set()

        input_images_amp = {}
        input_images_phase = {}
        reference_images_amp = {}
        reference_images_phase = {}

        for root, dirs, files in os.walk(root_path):
            if config.INPUT_FOLDER_NAME in root and config.AMP_FOLDER_NAME in root:
                folder_name = root.split('\\')[-3]
                folders.add(folder_name)
                input_images_amp[folder_name] = make_dataset(root, files)
            if config.INPUT_FOLDER_NAME in root and config.PHASE_FOLDER_NAME in root and phase:
                folder_name = root.split('\\')[-3]
                folders.add(folder_name)
                input_images_phase[folder_name] = make_dataset(root, files)
            if config.REFERENCE_FOLDER_NAME in root and config.AMP_FOLDER_NAME in root and training:
                folder_name = root.split('\\')[-3]
                folders.add(folder_name)
                reference_images_amp[folder_name] = make_dataset(root, files)
            if config.REFERENCE_FOLDER_NAME in root and config.PHASE_FOLDER_NAME in root and phase and training:
                folder_name = root.split('\\')[-3]
                folders.add(folder_name)
                reference_images_phase[folder_name] = make_dataset(root, files)

        self.input_images_amp = []
        self.input_images_phase = []
        self.reference_images_amp = []
        self.reference_images_phase = []

        for i in folders:
            self.input_images_amp.extend(input_images_amp[i])
            if phase:
                try:
                    self.input_images_phase.extend(input_images_phase[i])
                except KeyError:
                    print(f'Could not find input phase folder for {i} although phase was required')

            if training:
                assert (len(input_images_amp[i]) == len(reference_images_amp[i])), f'Not the same amount of input ' \
                                                                                   f'and ' \
                                                                                   f'reference images in amp folder ' \
                                                                                   f'{i}!'
                try:
                    self.reference_images_amp.extend(reference_images_amp[i])
                except KeyError:
                    print(f'Error, loader in training mode requires a reference folder for every input folder ({i})!')
                if phase:
                    assert (len(input_images_phase[i]) == len(reference_images_phase[i])), f'Not the same amount of ' \
                                                                                           f'input and reference ' \
                                                                                           f'images in phase folder ' \
                                                                                           f'{i}! '
                    try:
                        self.reference_images_phase.extend(reference_images_phase[i])
                    except KeyError:
                        print(f'Could not find reference phase folder for {i} although phase was required')

        self.phase = phase
        self.image_size = (image_size[1], image_size[2])
        self.training = training
        self.crops = crops
        self.val = False

    def _get_image(self, index: int, is_input: bool) -> np.ndarray:
        if is_input:
            amp_path = self.input_images_amp[index]
            if self.phase:
                phase_path = self.input_images_phase[index]
        else:
            amp_path = self.reference_images_amp[index]
            if self.phase:
                phase_path = self.reference_images_phase[index]
        amp = cv2.imread(amp_path, cv2.IMREAD_GRAYSCALE)

        if self.phase:
            phase = cv2.imread(phase_path, cv2.IMREAD_GRAYSCALE)
            amp_and_phase = np.stack([amp, phase])
            final_image = amp_and_phase
        else:
            final_image = amp

        return final_image

    def __getitem__(self, item: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        input_image = self._get_image(item, True)
        if input_image is None:
            print(self.input_images_amp[item])
        if self.training:
            reference_image = self._get_image(item, False)
            if self.phase:
                if input_image.shape[1] != reference_image.shape[1]:
                    input_image = input_image[:, input_image.shape[1] - reference_image.shape[1]:]
                combined = np.concatenate((input_image, reference_image), axis=0)
            else:
                combined = np.array([input_image, reference_image])
            if combined.shape[1] == self.image_size[0]:
                crop_x_lower = 0
            else:
                crop_x_lower = np.random.randint(0, combined.shape[1] - self.image_size[0])
            crop_x_upper = crop_x_lower + self.image_size[0]
            if combined.shape[2] == self.image_size[1]:
                crop_y_lower = 0
            else:
                crop_y_lower = np.random.randint(0, combined.shape[2] - self.image_size[1])
            crop_y_upper = crop_y_lower + self.image_size[1]
            if self.val:
                if input_image.shape[2] != 301:
                    lower = int((input_image.shape[2] - 301) / 2)
                    crops = ((0, 1024), (lower, lower + 301))
                else:
                    crops = ((0, 1024), (0, 301))
                combined_cropped = combined[:, crops[0][0]:crops[0][1], crops[1][0]:crops[1][1]]
            else:
                combined_cropped = combined[:, crop_x_lower:crop_x_upper, crop_y_lower:crop_y_upper]
            combined_cropped = np.moveaxis(combined_cropped, 0, -1)

            combined_PIL = ToPILImage()(combined_cropped)
            if self.val:
                transformed = ToTensor()(combined_PIL)
            else:
                transformed = self.transform(combined_PIL)
            if self.phase:
                i = 2
            else:
                i = 1
            input_image = transformed[0:i, :, :]
            reference_image = transformed[i:i + i, :, :]

            return input_image, reference_image
        else:
            if input_image.shape[2] != 301:
                lower = int((input_image.shape[2] - 301) / 2)
                crops = ((0, 1024), (lower, lower + 301))
            else:
                crops = ((0, 1024), (0, 301))
            if self.phase:
                cropped = input_image[:, crops[0][0]:crops[0][1], crops[1][0]:crops[1][1]]
                cropped = np.moveaxis(cropped, 0, -1)
            else:
                cropped = input_image[crops[0][0]:crops[0][1], crops[1][0]:crops[1][1]]
            cropped_PIL = ToPILImage()(cropped)
            input_image = self.transform(cropped_PIL)

            return input_image

    def __len__(self) -> int:
        return len(self.input_images_amp)
