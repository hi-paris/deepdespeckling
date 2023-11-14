import logging
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from glob import glob

from deepdespeckling.merlin.inference.model import Model
from deepdespeckling.utils.constants import M, m
from deepdespeckling.utils.utils import (denormalize_sar_image, load_sar_image, save_image_to_npy_and_png,
                                         symetrise_real_and_imaginary_parts, create_empty_folder_in_directory)


def save_despeckled_images(despeckled_images, image_name, save_dir):
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


def load_model(weights_path, patch_size):
    """Load model with given weights 

    Args:
        weights_path (str): path to weights  
        patch_size (int): patch size

    Returns:
        model (Model): model loaded with stored weights
    """
    model = Model(torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"), height=patch_size, width=patch_size)
    model.load_state_dict(torch.load(
        weights_path, map_location=torch.device('cpu')))

    return model


def initialize_axis_range(image_axis_dim, patch_size, stride_size):
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


def symetrise_real_and_imaginary_kernel(x, y, i_real_part, i_imag_part, patch_size):
    """ Get subpart of an image to denoise delimited by x and y as an imaginary and a real part, 
    symetrise it so that the noises are independant in each part,
    normalize it and return it as torch tensors

    Args:
        x (int): x image
        y (int): y image
        i_real_part (numpy array): real part of the image
        i_imag_part (numpy array): imaginary part of the image
        patch_size (int): patch size of the convolution

    Returns:
        images to denoise (torch tensor): real and imaginary parts symetrised as torch tensors
    """
    real_to_denoise, imag_to_denoise = symetrise_real_and_imaginary_parts(
        i_real_part[:, x:x + patch_size, y:y + patch_size, :], i_imag_part[:, x:x + patch_size, y:y + patch_size, :])

    real_to_denoise = torch.tensor(real_to_denoise).type(torch.float32)
    imag_to_denoise = torch.tensor(imag_to_denoise).type(torch.float32)

    real_to_denoise = (torch.log(torch.square(
        real_to_denoise)+1e-3)-2*m)/(2*(M-m))
    imag_to_denoise = (torch.log(torch.square(
        imag_to_denoise)+1e-3)-2*m)/(2*(M-m))

    return real_to_denoise, imag_to_denoise


def denoise_image_kernel(symetrised_noisy_image, symetrised_denoised_image, x, y, patch_size, model, normalisation_kernel=False):
    """Denoise a subpart of a given symetrised noisy image delimited by x, y and patch_size using a given model

    Args:
        symetrised_noisy_image (numpy array): symetrised noisy image to denoise
        symetrised_denoised_image (numpy array): symetrised partially denoised image
        x (int): x coordinate of current kernel to denoise
        y (int): y coordinate of current kernel to denoise
        patch_size (int): patch size
        model (Model): trained model with loaded weights 
        normalisation_kernel (bool, optional): Determine if. Defaults to False.

    Returns:
        symetrised_denoised_image (numpy array): image denoised in the given coordinates and the ones already iterated
    """
    if not normalisation_kernel:
        tmp_clean_image = model.forward(
            symetrised_noisy_image).detach().numpy()
        tmp_clean_image = np.moveaxis(tmp_clean_image, 1, -1)

        symetrised_denoised_image[:, x:x + patch_size, y:y + patch_size, :] = symetrised_denoised_image[:, x:x + patch_size,
                                                                                                        y:y + patch_size,
                                                                                                        :] + tmp_clean_image
    else:
        symetrised_denoised_image[:, x:x + patch_size, y:y + patch_size, :] = symetrised_denoised_image[:, x:x + patch_size,
                                                                                                        y:y + patch_size,
                                                                                                        :] + np.ones((1, patch_size, patch_size, 1))
    return symetrised_denoised_image


def preprocess_noisy_image(noisy_image):
    """preprocess a given noisy image and generates its real and imaginary parts

    Args:
        noisy_image (numpy array): noisy image

    Returns:
        noisy_image, noisy_image_real_part, noisy_image_imaginary_part (numpy array, numpy array, numpy array): 
        preprocessed noisy image, real part of noisy image, imaginary part of noisy image
    """
    noisy_image = np.array(noisy_image).reshape(
        1, np.size(noisy_image, 0), np.size(noisy_image, 1), 2)
    noisy_image_real_part = (noisy_image[:, :, :, 0]).reshape(noisy_image.shape[0], noisy_image.shape[1],
                                                              noisy_image.shape[2], 1)
    noisy_image_imaginary_part = (noisy_image[:, :, :, 1]).reshape(noisy_image.shape[0], noisy_image.shape[1],
                                                                   noisy_image.shape[2], 1)
    noisy_image = np.squeeze(
        np.sqrt(noisy_image_real_part ** 2 + noisy_image_imaginary_part ** 2))

    return noisy_image, noisy_image_real_part, noisy_image_imaginary_part


def preprocess_denoised_image(denoised_image_real_part, denoised_image_imaginary_part, count_image):
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


def denoise_image(noisy_image, weights_path, patch_size, stride_size):
    """Preprocess and denoise a coSAR image using given model weights

    Args:
        noisy_image (numpy array): numpy array containing the noisy image to despeckle 
        weights_path (str): path to the pth model file
        patch_size (int): size of the patch of the convolution
        stride_size (int): number of pixels between one convolution to the next

    Returns:
        output_image (numpy array): denoised image
    """
    noisy_image, noisy_image_real_part, noisy_image_imaginary_part = preprocess_noisy_image(
        noisy_image)

    # Pad the image
    image_height = np.size(noisy_image, 1)
    image_width = np.size(noisy_image, 2)

    model = load_model(
        weights_path=weights_path, patch_size=patch_size)

    count_image = np.zeros(noisy_image_real_part.shape)
    denoised_image_real_part = np.zeros(noisy_image_real_part.shape)
    denoised_image_imaginary_part = np.zeros(noisy_image_real_part.shape)

    x_range = initialize_axis_range(
        image_height, patch_size, stride_size)
    y_range = initialize_axis_range(
        image_width, patch_size, stride_size)

    for x in tqdm(x_range):
        for y in y_range:
            real_to_denoise, imag_to_denoise = symetrise_real_and_imaginary_kernel(
                x, y, noisy_image_real_part, noisy_image_imaginary_part, patch_size)

            denoised_image_real_part = denoise_image_kernel(
                real_to_denoise, denoised_image_real_part, x, y, patch_size, model)
            denoised_image_imaginary_part = denoise_image_kernel(
                imag_to_denoise, denoised_image_imaginary_part, x, y, patch_size, model)
            count_image = denoise_image_kernel(
                imag_to_denoise, count_image, x, y, patch_size, model, normalisation_kernel=True)

    denoised_image, denoised_image_real_part, denoised_image_imaginary_part = preprocess_denoised_image(
        denoised_image_real_part, denoised_image_imaginary_part, count_image)

    despeckled_image = {"noisy": {"full": noisy_image,
                                  "real": np.squeeze(noisy_image_real_part),
                                  "imaginary": np.squeeze(noisy_image_imaginary_part)
                                  },
                        "denoised": {"full": denoised_image,
                                     "real": denoised_image_real_part,
                                     "imaginary": denoised_image_imaginary_part
                                     }
                        }

    return despeckled_image


def denoise_images(images_to_denoise_path, weights_path, save_dir, patch_size, stride_size):
    """Iterate over a directory of coSAR images and store the denoised images in a directory

    Args:
        images_to_denoise_path (list): a list of paths of npy images to denoise
        weights_path (str): path to the pth file containing the weights of the model
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
        despeckled_images = denoise_image(
            noisy_image_idx, weights_path, patch_size, stride_size)

        logging.info(
            f"Saving despeckled images in {save_dir}")
        save_despeckled_images(
            despeckled_images, image_name, save_dir)
