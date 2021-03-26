# Installation

The easiest way to get the code running is to create a conda environment using the included *environment.yml*. After
that, one still needs to install the pytorch-fid package, which is not included in the file due to some installation
issues that might occur. Try running:

    pip install pytorch-fid

If that doesn't work, download the code from https://github.com/mseitzer/pytorch-fid and manually install the package.

If one wants to manually create an environment, the following packages, and their dependencies, should be installed:

* torch
* pytorch-lightning
* numpy
* piq
* matplotlib
* tensorboard
* torchsummary

# Training

For Training, the data needs to be in some folder, which is specified in the config.py file. The datasets inside can
also be split across multiple folders. Each dataset folder should contain one folder containing the input images and one folder containing the reference images.
The names of these folders can also be changed in config.py (default: "Full" for reference and "Gaped" for input). Each of these folders should contain at least one folder with the amplitude data and optionally a second one with the phase data (again names are configurable; default: "amp" and "phase").

The training configuration in general can be changed in config.py (including e.g. GPU usage etc.) and then executing the training is merely running the script.

Training can be monitored in tensorboard.

# Prediction

To use a trained model for predictions, one has to call predict.py using argument parsing to specify the different paths and options e.g.:

    python predict.py --model_path /path/to/model.ckpt --data_path /path/to/input/data --output_path /output/path [--phase --batch_size int --output_format]


# Evaluation

The evaluate.py script is used for calculating the metrics (FID, SSIM & PSNR) between the input and the reference images and the prediction and the reference images. It outputs both violin plots as well as statistics about the metrics.
    
    python evaluate.py /path/to/predicted /path/to/reference /path/to/input chart_title [--gpu]

# References

Network Architecture & part of loss function:

P. Isola, J. Zhu, T. Zhou, and A. A. Efros, “Image-to-image translation with conditional adversarial networks,” CoRR.
https://arxiv.org/abs/1611.07004v2 (2016).

Perceptual loss:

C. Wang, C. Xu, C. Wang, and D. Tao, “Perceptual adver-sarial networks for image-to-image transformation,” IEEE
Transactions on Image Process. 27, 4066–4079 https://arxiv.org/abs/1706.09138 (2018).

Frechét Inception Distance:

M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, andS. Hochreiter, “Gans trained by a two time-scale updaterule
converge to a local nash equilibrium,” Adv. Neural Inf.Process. Syst. 30, 6626–6637 https://arxiv.org/abs/1706.08500 (
2017).

and its PyTorch implementation: https://github.com/mseitzer/pytorch-fid