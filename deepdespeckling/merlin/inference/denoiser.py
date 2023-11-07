import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

from deepdespeckling.merlin.inference.model import Model
from deepdespeckling.utils.constants import M, m
from deepdespeckling.utils.utils import (denormalize_sar_for_testing, get_maximum_patch_size_from_image_dimensions, load_sar_images, save_real_imag_images,
                                         save_real_imag_images_noisy, save_sar_images, symetrisation_patch_test)


class Denoiser(object):
    """ Description
                ----------
                A set of initial conditions, and transformations on the Y

                Parameters
                ----------
                denoiser : an object

                Returns
                ----------
    """

    def __init__(self, input_c_dim=1):
        self.input_c_dim = input_c_dim

    def load(self, model, weights_path):
        """ Description
                    ----------
                    Restores a checkpoint located in a checkpoint repository

                    Parameters
                    ----------
                    checkpoint_dir : a path leading to the checkpoint file

                    Returns
                    ----------
                    True : Restoration is a success
                    False: Restoration has failed
        """
        print("[*] Loading the model...")

        model.load_state_dict(torch.load(weights_path))

        return model

    def image_to_real_and_imag(self, x, y, i_real_part, i_imag_part, patch_size):
        """ Get images to denoise in imaginary et real part as torch tensors

        Args:
            x (int): x image
            y (int): y image
            i_real_part (numpy array): real part of the image
            i_imag_part (numpy array): imaginary part of the image
            patch_size (int): patch size of the convolution

        Returns:
            images to denoise (torch tensor): real and imaginary parts as torch tensors
        """
        real_to_denoise, imag_to_denoise = symetrisation_patch_test(
            i_real_part[:, x:x + patch_size, y:y + patch_size, :], i_imag_part[:, x:x + patch_size, y:y + patch_size, :])

        real_to_denoise = torch.tensor(real_to_denoise)
        imag_to_denoise = torch.tensor(imag_to_denoise)

        real_to_denoise = real_to_denoise.type(torch.float32)
        imag_to_denoise = imag_to_denoise.type(torch.float32)

        real_to_denoise = (torch.log(torch.square(
            real_to_denoise)+1e-3)-2*m)/(2*(M-m))
        imag_to_denoise = (torch.log(torch.square(
            imag_to_denoise)+1e-3)-2*m)/(2*(M-m))

        return real_to_denoise, imag_to_denoise

    def denoise_image(self, image_path, weights_path, save_dir, stride):
        """Denoise a coSAR image and store the results in a given directory

        Args:
            image_path (str): path to the npy image file
            weights_path (str): path to the pth model file
            save_dir (str): path to the directory where the results will be store
            stride (int): stride of the convolution

        Returns:
            output_image (numpy array): denoised image
        """
        real_image = load_sar_images(image_path).astype(np.float32)
        i_real_part = (real_image[:, :, :, 0]).reshape(real_image.shape[0], real_image.shape[1],
                                                       real_image.shape[2], 1)
        i_imag_part = (real_image[:, :, :, 1]).reshape(real_image.shape[0], real_image.shape[1],
                                                       real_image.shape[2], 1)

        # Pad the image
        im_h = np.size(real_image, 1)
        im_w = np.size(real_image, 2)

        patch_size = get_maximum_patch_size_from_image_dimensions(
            kernel_size=256, height=im_h, width=im_w)
        print(f"The model uses a patch size of {patch_size}")

        loaded_model = Model(torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"), height=patch_size, width=patch_size)
        loaded_model.load_state_dict(torch.load(
            weights_path, map_location=torch.device('cpu')))

        count_image = np.zeros(i_real_part.shape)
        output_clean_image_1 = np.zeros(i_real_part.shape)
        output_clean_image_2 = np.zeros(i_real_part.shape)

        if im_h == patch_size:
            x_range = list(np.array([0]))
        else:
            x_range = list(range(0, im_h - patch_size, stride))
            if (x_range[-1] + patch_size) < im_h:
                x_range.extend(range(im_h - patch_size, im_h - patch_size + 1))

        if im_w == patch_size:
            y_range = list(np.array([0]))
        else:
            y_range = list(range(0, im_w - patch_size, stride))
            if (y_range[-1] + patch_size) < im_w:
                y_range.extend(range(im_w - patch_size, im_w - patch_size + 1))

        for x in tqdm(x_range):
            for y in y_range:
                real_to_denoise, imag_to_denoise = self.image_to_real_and_imag(
                    x, y, i_real_part, i_imag_part, patch_size)

                tmp_clean_image_real = loaded_model.forward(
                    real_to_denoise).detach().numpy()
                tmp_clean_image_real = np.moveaxis(tmp_clean_image_real, 1, -1)

                output_clean_image_1[:, x:x + patch_size, y:y + patch_size, :] = output_clean_image_1[:, x:x + patch_size,
                                                                                                      y:y + patch_size,
                                                                                                      :] + tmp_clean_image_real

                tmp_clean_image_imag = loaded_model.forward(
                    imag_to_denoise).detach().numpy()
                tmp_clean_image_imag = np.moveaxis(tmp_clean_image_imag, 1, -1)

                output_clean_image_2[:, x:x + patch_size, y:y + patch_size, :] = output_clean_image_2[:, x:x + patch_size,
                                                                                                      y:y + patch_size,
                                                                                                      :] + tmp_clean_image_imag
                count_image[:, x:x + patch_size, y:y + patch_size, :] = count_image[:, x:x + patch_size, y:y + patch_size,
                                                                                    :] + np.ones((1, patch_size, patch_size, 1))

        output_clean_image_1 = output_clean_image_1 / count_image
        output_clean_image_2 = output_clean_image_2 / count_image

        # combine the two estimation
        output_clean_image = 0.5 * (np.square(denormalize_sar_for_testing(
            output_clean_image_1)) + np.square(denormalize_sar_for_testing(output_clean_image_2)))

        noisyimage = np.squeeze(np.sqrt(i_real_part ** 2 + i_imag_part ** 2))
        output_image = np.sqrt(np.squeeze(output_clean_image))

        imagename = Path(image_path).name

        print(f"Denoised image {imagename}")

        save_sar_images(output_image, noisyimage, imagename, save_dir)
        save_real_imag_images(noisyimage, denormalize_sar_for_testing(output_clean_image_1), denormalize_sar_for_testing(output_clean_image_2),
                              imagename, save_dir)
        save_real_imag_images_noisy(noisyimage, np.squeeze(
            i_real_part), np.squeeze(i_imag_part), imagename, save_dir)

        return output_image

    def denoise_images(self, images_to_denoise_paths, weights_path, save_dir, stride):
        """Iterate over a directory of coSAR images and store the denoised images in a directory

        Args:
            images_to_denoise_paths (list): a list of paths of npy images to denoise
            weights_path (str): path to the pth file containing the weights of the model
            save_dir (str): repository to save sar images, real images and noisy images
            stride (int): number of bytes from one row of pixels in memory to the next row of pixels in memory
            patch_size (int): size of the patch of the convolution

        Returns:
            idx_image (numpya array) : last image to be denoised
        """

        assert len(images_to_denoise_paths) != 0, 'No data!'

        print(f"Starting denoising images in {images_to_denoise_paths}")

        for idx in range(len(images_to_denoise_paths)):
            idx_image = self.denoise_image(
                images_to_denoise_paths[idx], weights_path, save_dir, stride)

        return idx_image
