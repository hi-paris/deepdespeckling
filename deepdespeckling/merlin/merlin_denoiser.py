from glob import glob
import logging
from pathlib import Path
import torch
import os
import numpy as np
from tqdm import tqdm

from deepdespeckling.denoiser import Denoiser
from deepdespeckling.model import Model
from deepdespeckling.utils.constants import M, m
from deepdespeckling.utils.utils import (denormalize_sar_image, load_sar_image, save_image_to_npy_and_png,
                                         symetrise_real_and_imaginary_parts, create_empty_folder_in_directory)

current_dir = os.path.dirname(__file__)


class MerlinDenoiser(Denoiser):
    """Class to share parameters beyond denoising functions
    """

    def __init__(self, model_name, symetrise, **params):
        """Initialize MerlinDenoiser class

        Args:
            model_name (str): name to be used, can be "spotlight" or "stripmap"
        """
        super().__init__(**params)
        self.model_name = model_name
        self.symetrise = symetrise
        self.weights_path = self.init_model_weights_path()

    def init_model_weights_path(self) -> str:
        """Get model weights path from model name

        Returns:
            model_weights_path (str): the path of the weights of the specified model
        """
        if self.model_name == "spotlight":
            model_weights_path = os.path.join(
                current_dir, "saved_models/spotlight.pth")
        elif self.model_name == "stripmap":
            model_weights_path = os.path.join(
                current_dir, "saved_models/stripmap.pth")
        else:
            raise ValueError(
                "The model name doesn't refer to an existing model ")

        return model_weights_path

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
            self.weights_path, map_location=torch.device("cpu")))

        return model

    def save_despeckled_images(self, despeckled_images: dict, image_name: str, save_dir: str):
        """Save full, real and imaginary part of noisy and denoised image stored in a dictionary in png to a given folder

        Args:
            despeckled_images (dict): dictionary containing full, real and imaginary parts of noisy and denoised image
            image_name (str): name of the image
            save_dir (str): path to the folder where to save the png images
        """
        threshold = np.mean(
            despeckled_images["noisy"]["full"]) + 3 * np.std(despeckled_images["noisy"]["full"])
        image_name = image_name.split('\\')[-1]

        for key in despeckled_images:
            create_empty_folder_in_directory(save_dir, key)
            for key2 in despeckled_images[key]:
                save_image_to_npy_and_png(
                    despeckled_images[key][key2], save_dir, f"/{key}/{key}_{key2}_", image_name, threshold)

    def denoise_image_kernel(self, noisy_image: torch.tensor, denoised_image: np.array, x: int, y: int, patch_size: int,
                             model: Model, normalisation_kernel: bool = False) -> np.array:
        """Denoise a subpart of a given symetrised noisy image delimited by x, y and patch_size using a given model

        Args:
            noisy_image (torch tensor): symetrised noisy image to denoise
            denoised_image (numpy array): symetrised partially denoised image
            x (int): x coordinate of current kernel to denoise
            y (int): y coordinate of current kernel to denoise
            patch_size (int): patch size
            model (Model): trained model with loaded weights 
            normalisation_kernel (bool, optional): Determine if. Defaults to False.

        Returns:
            denoised_image (numpy array): image denoised in the given coordinates and the ones already iterated
        """
        if not normalisation_kernel:

            if self.device != 'cpu':
                tmp_clean_image = model.forward(
                    noisy_image).cpu().detach().numpy()
            else:
                tmp_clean_image = model.forward(
                    noisy_image).detach().numpy()

            tmp_clean_image = np.moveaxis(tmp_clean_image, 1, -1)
            denoised_image[:, x:x + patch_size, y:y + patch_size, :] = denoised_image[:, x:x + patch_size,
                                                                                      y:y + patch_size,
                                                                                      :] + tmp_clean_image
        else:
            denoised_image[:, x:x + patch_size, y:y + patch_size, :] = denoised_image[:, x:x + patch_size,
                                                                                      y:y + patch_size,
                                                                                      :] + np.ones((1, patch_size, patch_size, 1))
        return denoised_image

    def preprocess_noisy_image(self, noisy_image: np.array) -> tuple[np.array, np.array, np.array]:
        """preprocess a given noisy image and generates its real and imaginary parts

        Args:
            noisy_image (numpy array): noisy image

        Returns:
            noisy_image, noisy_image_real_part, noisy_image_imaginary_part (numpy array, numpy array, numpy array): 
            preprocessed noisy image, real part of noisy image, imaginary part of noisy image
        """
        noisy_image_real_part = (noisy_image[:, :, :, 0]).reshape(noisy_image.shape[0], noisy_image.shape[1],
                                                                  noisy_image.shape[2], 1)
        noisy_image_imaginary_part = (noisy_image[:, :, :, 1]).reshape(noisy_image.shape[0], noisy_image.shape[1],
                                                                       noisy_image.shape[2], 1)
        noisy_image = np.squeeze(
            np.sqrt(noisy_image_real_part ** 2 + noisy_image_imaginary_part ** 2))

        return noisy_image, noisy_image_real_part, noisy_image_imaginary_part

    def preprocess_denoised_image(self, denoised_image_real_part: np.array, denoised_image_imaginary_part: np.array, count_image: np.array) -> tuple[np.array, np.array, np.array]:
        """Preprocess given denoised real and imaginary parts of an image, and build the full denoised image

        Args:
            denoised_image_real_part (numpy array): real part of a denoised image
            denoised_image_imaginary_part (numpy array): imaginary part of a denoised image
            count_image (numpy array): normalisation image used for denormalisation

        Returns:
            denoised_image, denoised_image_real_part, denoised_image_imaginary_part (numpy array, numpy array, numpy array): 
            processed denoised full image, processed denoised image real part, processed denoised image imaginary part
        """
        denoised_image_real_part = denormalize_sar_image(
            denoised_image_real_part / count_image)
        denoised_image_imaginary_part = denormalize_sar_image(
            denoised_image_imaginary_part / count_image)

        # combine the two estimation
        output_clean_image = 0.5 * (np.square(
            denoised_image_real_part) + np.square(denoised_image_imaginary_part))

        denoised_image = np.sqrt(np.squeeze(output_clean_image))

        return denoised_image, denoised_image_real_part, denoised_image_imaginary_part

    def denoise_image(self, noisy_image: np.array, patch_size: int, stride_size: int) -> dict:
        """Preprocess and denoise a coSAR image using given model weights

        Args:
            noisy_image (numpy array): numpy array containing the noisy image to despeckle 
            patch_size (int): size of the patch of the convolution
            stride_size (int): number of pixels between one convolution to the next

        Returns:
            despeckled_image (dict): noisy and denoised images
        """
        noisy_image = np.array(noisy_image).reshape(
            1, np.size(noisy_image, 0), np.size(noisy_image, 1), 2)

        # Pad the image
        image_height = np.size(noisy_image, 1)
        image_width = np.size(noisy_image, 2)

        noisy_image, noisy_image_real_part, noisy_image_imaginary_part = self.preprocess_noisy_image(
            noisy_image)

        model = self.load_model(patch_size=patch_size)

        count_image = np.zeros(noisy_image_real_part.shape)
        denoised_image_real_part = np.zeros(noisy_image_real_part.shape)
        denoised_image_imaginary_part = np.zeros(noisy_image_real_part.shape)

        x_range = self.initialize_axis_range(
            image_height, patch_size, stride_size)
        y_range = self.initialize_axis_range(
            image_width, patch_size, stride_size)

        for x in tqdm(x_range):
            for y in y_range:
                real_to_denoise = noisy_image_real_part[:,
                                                        x:x + patch_size, y:y + patch_size, :]
                imag_to_denoise = noisy_image_imaginary_part[:,
                                                             x:x + patch_size, y:y + patch_size, :]
                if self.symetrise:
                    real_to_denoise, imag_to_denoise = symetrise_real_and_imaginary_parts(
                        real_to_denoise, imag_to_denoise)

                real_to_denoise = torch.tensor(
                    real_to_denoise, device=self.device, dtype=torch.float32)
                imag_to_denoise = torch.tensor(
                    imag_to_denoise, device=self.device, dtype=torch.float32)

                real_to_denoise = (torch.log(torch.square(
                    real_to_denoise)+1e-3)-2*m)/(2*(M-m))
                imag_to_denoise = (torch.log(torch.square(
                    imag_to_denoise)+1e-3)-2*m)/(2*(M-m))

                denoised_image_real_part = self.denoise_image_kernel(
                    real_to_denoise, denoised_image_real_part, x, y, patch_size, model)
                denoised_image_imaginary_part = self.denoise_image_kernel(
                    imag_to_denoise, denoised_image_imaginary_part, x, y, patch_size, model)
                count_image = self.denoise_image_kernel(
                    imag_to_denoise, count_image, x, y, patch_size, model, normalisation_kernel=True)

        denoised_image, denoised_image_real_part, denoised_image_imaginary_part = self.preprocess_denoised_image(
            denoised_image_real_part, denoised_image_imaginary_part, count_image)

        despeckled_image = {"noisy": {"full": noisy_image,
                                      "real": np.squeeze(noisy_image_real_part),
                                      "imaginary": np.squeeze(noisy_image_imaginary_part)
                                      },
                            "denoised": {"full": denoised_image,
                                         "from_real": denoised_image_real_part,
                                         "from_imaginary": denoised_image_imaginary_part
                                         }
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
