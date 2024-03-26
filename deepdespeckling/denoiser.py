import logging
import torch
import numpy as np


class Denoiser:
    """Class to share parameters beyond denoising functions
    """

    def __init__(self):
        self.device = self.get_device()

    def get_device(self) -> str:
        """Get torch device to use depending on gpu's availability

        Returns:
            device (str): device to be used by torch
        """
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
        logging.info(f"{device} device is used by torch")

        return device

    def initialize_axis_range(self, image_axis_dim: int, patch_size: int, stride_size: int) -> list:
        """Initialize the convolution range for x or y axis

        Args:
            image_axis_dim (int): axis size
            patch_size (int): patch size
            stride_size (int): stride size

        Returns:
            axis_range (list) : pixel borders of each convolution
        """
        if image_axis_dim == patch_size:
            axis_range = list(np.array([0]))
        else:
            axis_range = list(
                range(0, image_axis_dim - patch_size, stride_size))
            if (axis_range[-1] + patch_size) < image_axis_dim:
                axis_range.extend(
                    range(image_axis_dim - patch_size, image_axis_dim - patch_size + 1))

        return axis_range

    def save_despeckled_images(self):
        raise NotImplementedError

    def denoise_image_kernel(self):
        raise NotImplementedError

    def preprocess_denoised_image(self):
        raise NotImplementedError

    def denoise_image(self):
        raise NotImplementedError

    def denoise_images(self):
        raise NotImplementedError
