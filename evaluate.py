import os
import cv2
import torch
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from utils import fid
from piq import ssim, psnr
from argparse import ArgumentParser
from torchvision.transforms import ToPILImage, ToTensor


def stretch_contrast(img: np.ndarray, max_val: int, min_val: int) -> np.ndarray:
    return (img - min_val) * (255 / (max_val - min_val)).astype(np.uint8)


def read_folder(paths: list, forged_shape: tuple = None) -> list:
    result = []
    arrays = []
    for i in sorted(paths):
        image_raw = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        if forged_shape:
            lower = int((image_raw.shape[1] - forged_shape[2]) / 2)
            image_cropped = image_raw[0:forged_shape[1], lower:lower + forged_shape[2]]
        else:
            image_cropped = image_raw
        arrays.append(image_cropped)
        image_PIL = ToPILImage()(image_cropped)
        image_tensor = ToTensor()(image_PIL)
        result.append(image_tensor)

    cv2.imshow("Contrast", stretch_contrast(arrays[0], np.max(arrays), np.min(arrays)))
    cv2.waitKey()
    return result


def plot_comparison(input_ssim: pd.DataFrame,
                    input_psnr: pd.DataFrame,
                    forged_ssim: pd.DataFrame,
                    forged_psnr: pd.DataFrame,
                    title: str) -> None:
    ssim_df = input_ssim.append(forged_ssim)
    psnr_df = input_psnr.append(forged_psnr)
    fig, ax = plt.subplots(2, 1, gridspec_kw={"wspace": 1}, figsize=[6.4, 9.6])
    fig.suptitle(title, fontsize=24)
    sns.violinplot(x="value", y="dataset", data=ssim_df, ax=ax[0])
    ax[0].set_title("SSIMs")
    sns.violinplot(x="value", y="dataset", data=psnr_df, ax=ax[1])
    ax[1].set_title("PSNRs")
    plt.show()


def plot_individual(input_ssim: pd.DataFrame,
                    input_psnr: pd.DataFrame,
                    forged_ssim: pd.DataFrame,
                    forged_psnr: pd.DataFrame,
                    title: str) -> None:
    fig, ax = plt.subplots(4, 1, gridspec_kw={"wspace": 1}, figsize=[6.4, 4 * 4.8])
    fig.suptitle(title, fontsize=24)
    sns.violinplot(x="value", y="dataset", data=input_ssim, ax=ax[0])
    ax[0].set_title("SSIMs input")
    sns.violinplot(x="value", y="dataset", data=forged_ssim, ax=ax[1])
    ax[1].set_title("SSIMs forged")
    sns.violinplot(x="value", y="dataset", data=input_psnr, ax=ax[2])
    ax[2].set_title("PSNRs input")
    sns.violinplot(x="value", y="dataset", data=forged_psnr, ax=ax[3])
    ax[3].set_title("PSNRs forged")
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('forged_path')
    parser.add_argument('reference_path')
    parser.add_argument('input_path')
    parser.add_argument('title')
    parser.add_argument('--gpu', dest='use_gpu', action='store_const', const=True, default=False)
    args = parser.parse_args()

    if args.use_gpu:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    forged_paths = []
    reference_paths = []
    input_paths = []

    for i in os.scandir(args.forged_path):
        if (i.path.endswith('.tiff') or i.path.endswith('.tif')) and i.is_file():
            forged_paths.append(i.path)

    for i in os.scandir(args.reference_path):
        if (i.path.endswith('.tiff') or i.path.endswith('.tif')) and i.is_file():
            reference_paths.append(i.path)

    for i in os.scandir(args.input_path):
        if (i.path.endswith('.tiff') or i.path.endswith('.tif')) and i.is_file():
            input_paths.append(i.path)

    forged = read_folder(forged_paths)
    reference = read_folder(reference_paths, forged[0].shape)
    inputs = read_folder(input_paths, forged[0].shape)

    forged_tensor = torch.cat(forged, 0).reshape((len(inputs), 1, inputs[0].shape[1], inputs[0].shape[2]))
    reference_tensor = torch.cat(reference, 0).reshape(len(inputs), 1, inputs[0].shape[1], inputs[0].shape[2])
    inputs_tensor = torch.cat(inputs, 0).reshape((len(inputs), 1, inputs[0].shape[1], inputs[0].shape[2]))

    forged_tensor = forged_tensor.expand(-1, 3, -1, -1).clone().to(device)
    reference_tensor = reference_tensor.expand(-1, 3, -1, -1).clone().to(device)
    inputs_tensor = inputs_tensor.expand(-1, 3, -1, -1).clone().to(device)

    forged_tensor *= 255
    reference_tensor *= 255
    inputs_tensor *= 255

    fid_forged = fid(forged_tensor, reference_tensor, 192, torch.device(device))
    fid_input = fid(inputs_tensor, reference_tensor, 192, torch.device(device))

    print(f"FID Score input: {fid_input}")
    print(f"FID Score forged: {fid_forged}")

    ssims_forged = {}
    psnrs_forged = {}
    ssims_input = {}
    psnrs_input = {}
    ssims_forged["value"] = []
    psnrs_forged["value"] = []
    ssims_input["value"] = []
    psnrs_input["value"] = []
    ssims_forged["dataset"] = []
    psnrs_forged["dataset"] = []
    ssims_input["dataset"] = []
    psnrs_input["dataset"] = []

    for i, j, k in zip(forged, reference, inputs):
        ssims_forged["value"].append(ssim(i, j, data_range=1.).item())
        psnrs_forged["value"].append(psnr(i, j, data_range=1.).item())
        ssims_input["value"].append(ssim(j, k, data_range=1.).item())
        psnrs_input["value"].append(psnr(j, k, data_range=1.).item())
        ssims_forged["dataset"].append("forged")
        psnrs_forged["dataset"].append("forged")
        ssims_input["dataset"].append("input")
        psnrs_input["dataset"].append("input")

    ssims_input_df = pd.DataFrame(ssims_input)
    ssims_forged_df = pd.DataFrame(ssims_forged)
    psnrs_input_df = pd.DataFrame(psnrs_input)
    psnrs_forged_df = pd.DataFrame(psnrs_forged)

    plot_comparison(ssims_input_df, psnrs_input_df, ssims_forged_df, psnrs_forged_df, args.title)
    plot_individual(ssims_input_df, psnrs_input_df, ssims_forged_df, psnrs_forged_df, args.title)
    print("SSIM input statistics:")
    print(ssims_input_df["value"].describe())
    print("SSIM forged statistics:")
    print(ssims_forged_df["value"].describe())
    print("PSNR input statistics:")
    print(psnrs_input_df["value"].describe())
    print("PSNR forged statistics:")
    print(psnrs_forged_df["value"].describe())
