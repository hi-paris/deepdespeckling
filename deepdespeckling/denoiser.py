import logging
import torch
import numpy as np
from pathlib import Path
from glob import glob

from deepdespeckling.model import Model
from deepdespeckling.utils.utils import load_sar_image


class Denoiser():
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

    def build_dynamic_parameters_dict(self, noisy_image: np.array, weights_path: str, patch_size: int, stride_size: int, symetrise: bool = True) -> dict:
        """build a dictionary of parameters for denoise_images function excluding symetrise parameter 
            if using Sar2SarDenoiser

        Args:
            noisy_image (numpy array): numpy array containing the noisy image to despeckle 
            weights_path (str): path to the pth model file
            patch_size (int): size of the patch of the convolution
            stride_size (int): number of pixels between one convolution to the next
            symetrise (bool, optional): _description_. Defaults to True.

        Returns:
            dict: dictionary of parameters for denoise_images function
        """

        dict_params = {
            "noisy_image": noisy_image,
            "weights_path": weights_path,
            "patch_size": patch_size,
            "stride_size": stride_size,
            "symetrise": symetrise
        }

        if self.__class__.__name__ == "Sar2SarDenoiser":
            del dict_params['symetrise']

        return dict_params

    def denoise_images(self, images_to_denoise_path: list, weights_path: str, save_dir: str, patch_size: int,
                       stride_size: int, symetrise: bool = True):
        """Iterate over a directory of coSAR images and store the denoised images in a directory

        Args:
            images_to_denoise_path (list): a list of paths of npy images to denoise
            weights_path (str): path to the pth file containing the weights of the model
            save_dir (str): repository to save sar images, real images and noisy images
            patch_size (int): size of the patch of the convolution
            stride_size (int): number of pixels between one convolution to the next
            symetrise (bool) : if using MerlinDenoiser, if True, will symetrise the real and 
                imaginary parts of the noisy image. Defaults to True
        """

        images_to_denoise_paths = glob((images_to_denoise_path + '/*.npy'))

        assert len(images_to_denoise_paths) != 0, 'No data!'

        logging.info(f"Starting denoising images in {images_to_denoise_paths}")

        for idx in range(len(images_to_denoise_paths)):
            image_name = Path(images_to_denoise_paths[idx]).name
            logging.info(
                f"Despeckling {image_name}")

            noisy_image_idx = load_sar_image(
                images_to_denoise_paths[idx]).astype(np.float32)
            dict_params = self.build_dynamic_parameters_dict(
                noisy_image_idx, weights_path, patch_size, stride_size, symetrise)
            despeckled_images = self.denoise_image(**dict_params)

            logging.info(
                f"Saving despeckled images in {save_dir}")
            self.save_despeckled_images(
                despeckled_images, image_name, save_dir)
