import logging
import os
import torch
import numpy as np

from deepdespeckling.model import Model

this_dir, _ = os.path.split(__file__)


class Denoiser:
    """Class to share parameters beyond denoising functions
    """

    def __init__(self):
        self.device = self.get_device()

    def get_model_weights_path(self, model_name: str) -> str:
        """Get model weights path from model name

        Args:
            model_name (str): model name, either "spotlight" or "stripmap" to select MERLIN model on the 
                right cosar image format or "sar2sar" for SAR2SAR model

        Returns:
            model_weights_path (str): the path of the weights of the specified model
        """
        if model_name == "spotlight":
            model_weights_path = os.path.join(
                this_dir, "merlin/saved_model", "spotlight.pth")
        elif model_name == "stripmap":
            model_weights_path = os.path.join(
                this_dir, "merlin/saved_model", "stripmap.pth")
        elif model_name == "sar2sar":
            model_weights_path = os.path.join(
                this_dir, "sar2sar/saved_model", "sar2sar.pth")
        else:
            raise ValueError(
                "The model name doesn't refer to an existing model ")

        return model_weights_path

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

    def load_model(self, weights_path: str, patch_size: int) -> Model:
        """Load model with given weights 

        Args:
            weights_path (str): path to weights  
            patch_size (int): patch size

        Returns:
            model (Model): model loaded with stored weights
        """
        model = Model(torch.device(self.device),
                      height=patch_size, width=patch_size)
        model.load_state_dict(torch.load(
            weights_path, map_location=torch.device("cpu")))

        return model

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
