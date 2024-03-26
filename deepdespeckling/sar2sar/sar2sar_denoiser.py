from glob import glob
import logging
import os
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

from deepdespeckling.denoiser import Denoiser
from deepdespeckling.model import Model
from deepdespeckling.utils.constants import M, m
from deepdespeckling.utils.utils import (denormalize_sar_image, load_sar_image, normalize_sar_image, save_image_to_npy_and_png,
                                         create_empty_folder_in_directory)

current_dir = os.path.dirname(__file__)


class Sar2SarDenoiser(Denoiser):
    """Class to share parameters beyond denoising functions
    """

    def __init__(self, **params):
        super().__init__(**params)
        self.weights_path = os.path.join(
            current_dir, "saved_model/sar2sar.pth")
        print(self.weights_path)

    def load_model(self, patch_size: int) -> Model:
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
            self.weights_path, map_location=torch.device("cpu"))['model_state_dict'])

        return model

    def save_despeckled_images(self, despeckled_images: dict, image_name: str, save_dir: str):
        """Save full, real and imaginary part of noisy and denoised image stored in a dictionary in png to a given folder

        Args:
            despeckled_images (dict): dictionary containing noisy and denoised image
            image_name (str): name of the image
            save_dir (str): path to the folder where to save the png images
        """
        threshold = np.mean(
            despeckled_images["noisy"]) + 3 * np.std(despeckled_images["noisy"])
        image_name = image_name.split('\\')[-1]

        for key in despeckled_images:
            create_empty_folder_in_directory(save_dir, key)
            save_image_to_npy_and_png(
                despeckled_images[key], save_dir, f"/{key}/{key}_", image_name, threshold)

    def denoise_image_kernel(self, noisy_image_kernel: torch.tensor, denoised_image_kernel: np.array, x: int, y: int, patch_size: int, model: Model, normalisation_kernel: bool = False) -> np.array:
        """Denoise a subpart of a given symetrised noisy image delimited by x, y and patch_size using a given model

        Args:
            noisy_image_kernel (torch tensor): part of the noisy image to denoise 
            denoised_image_kernel (numpy array): part of the partially denoised image
            x (int): x coordinate of current kernel to denoise
            y (int): y coordinate of current kernel to denoise
            patch_size (int): patch size
            model (Model): trained model with loaded weights 
            normalisation_kernel (bool, optional): Determine if. Defaults to False.

        Returns:
            denoised_image_kernel (numpy array): image denoised in the given coordinates and the ones already iterated
        """
        if not normalisation_kernel:

            with torch.no_grad():
                if self.device != 'cpu':
                    tmp_clean_image = model.forward(
                        noisy_image_kernel).cpu().numpy()
                else:
                    tmp_clean_image = model.forward(
                        noisy_image_kernel).numpy()

            tmp_clean_image = denormalize_sar_image(np.squeeze(
                np.asarray(tmp_clean_image)))

            denoised_image_kernel[x:x + patch_size, y:y + patch_size] = denoised_image_kernel[x:x +
                                                                                              patch_size, y:y + patch_size] + tmp_clean_image
        else:
            denoised_image_kernel[x:x + patch_size, y:y + patch_size] = denoised_image_kernel[x:x +
                                                                                              patch_size, y:y + patch_size] + np.ones((patch_size, patch_size))

        return denoised_image_kernel

    def denormalize_sar_image(self, image: np.array) -> np.array:
        """Denormalize a sar image stored in a numpy array

        Args:
            image (numpy array): a sar image

        Raises:
            TypeError: raise an error if the image file is not a numpy array

        Returns:
            (numpy array): the image denormalized
        """
        if not isinstance(image, np.ndarray):
            raise TypeError('Please provide a numpy array')
        return np.exp((np.clip(np.squeeze(image), 0, image.max()))*(M-m)+m)

    def denoise_image(self, noisy_image: np.array, patch_size: int, stride_size: int) -> dict:
        """Preprocess and denoise a coSAR image using given model weights

        Args:
            noisy_image (numpy array): numpy array containing the noisy image to despeckle 
            patch_size (int): size of the patch of the convolution
            stride_size (int): number of pixels between one convolution to the next

        Returns:
            output_image (numpy array): denoised image
        """
        noisy_image = np.array(noisy_image).reshape(
            1, np.size(noisy_image, 0), np.size(noisy_image, 1), 1).astype(np.float32)

        noisy_image = normalize_sar_image(noisy_image)

        noisy_image = torch.tensor(
            noisy_image, dtype=torch.float)

        # Pad the image
        image_height = noisy_image.size(dim=1)
        image_width = noisy_image.size(dim=2)

        model = self.load_model(patch_size=patch_size)

        count_image = np.zeros((image_height, image_width))
        denoised_image = np.zeros((image_height, image_width))

        x_range = self.initialize_axis_range(
            image_height, patch_size, stride_size)
        y_range = self.initialize_axis_range(
            image_width, patch_size, stride_size)

        for x in tqdm(x_range):
            for y in y_range:
                noisy_image_kernel = noisy_image[:,
                                                 x:x + patch_size, y:y + patch_size, :]
                noisy_image_kernel = noisy_image_kernel.to(self.device)

                denoised_image = self.denoise_image_kernel(
                    noisy_image_kernel, denoised_image, x, y, patch_size, model)
                count_image = self.denoise_image_kernel(
                    noisy_image_kernel, count_image, x, y, patch_size, model, normalisation_kernel=True)

        denoised_image = denoised_image / count_image

        noisy_image_denormalized = self.denormalize_sar_image(
            np.squeeze(np.asarray(noisy_image.cpu().numpy())))

        despeckled_image = {"noisy": noisy_image_denormalized,
                            "denoised": denoised_image
                            }

        return despeckled_image

    def denoise_images(self, images_to_denoise_path: list, save_dir: str, patch_size: int,
                       stride_size: int):
        """Iterate over a directory of coSAR images and store the denoised images in a directory

        Args:
            images_to_denoise_path (list): a list of paths of npy images to denoise
            save_dir (str): repository to save sar images, real images and noisy images
            patch_size (int): size of the patch of the convolution
            stride_size (int): number of pixels between one convolution to the next
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
            despeckled_images = self.denoise_image(
                noisy_image_idx, patch_size, stride_size)

            logging.info(
                f"Saving despeckled images in {save_dir}")
            self.save_despeckled_images(
                despeckled_images, image_name, save_dir)
