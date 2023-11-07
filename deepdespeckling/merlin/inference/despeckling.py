import logging
import os
from glob import glob
from pathlib import Path
import numpy as np

from deepdespeckling.merlin.inference.denoiser import Denoiser
from deepdespeckling.utils.utils import (crop, crop_fixed, get_info_image, preprocess_and_store_sar_images_from_coordinates,
                                         store_data_and_plot, create_empty_folder_in_directory,
                                         preprocess_and_store_sar_images)


this_dir, this_filename = os.path.split(__file__)


def despeckle(sar_images_path, destination_directory_path, stride_size=64, model_weights_path=os.path.join(this_dir, "saved_model", "spotlight.pth"), patch_size=256):
    """Despeckle coSAR images using trained MERLIN (spotlight or stripmap weights)

    Args:
        sar_images_path (str): path of sar images
        destination_directory_path (str): path of folder in which results will be stored
        stride_size (int): stride size. Defaults to 64.
        model_weights_path (str): path to model weights (pth file). Defaults to os.path.join(this_dir, "saved_model", "spotlight.pth").
        patch_size (int): patch size. Defaults to 256.

    Returns:
        denoised_image (npy): denoised image in a numpy array (last image contained in sar_images_path)
    """

    logging.info(
        f"""Despeckling entire images using {model_weights_path.split("/")[-1]} weights""")

    processed_images_path = create_empty_folder_in_directory(destination_directory_path=destination_directory_path,
                                                             folder_name="processed_images")
    preprocess_and_store_sar_images(
        sar_images_path=sar_images_path, processed_images_path=processed_images_path)

    processed_images_paths = glob((processed_images_path + '/*.npy'))

    logging.info(
        f"Starting inference.. Working directory: {os.getcwd()}. Collecting data from {sar_images_path} and storing test results in {destination_directory_path}")
    denoised_image = Denoiser().denoise_images(images_to_denoise_paths=processed_images_paths, weights_path=model_weights_path, save_dir=destination_directory_path,
                                               stride=stride_size, patch_size=patch_size)

    return denoised_image


def despeckle_from_coordinates(sar_images_path, coordinates_dict, destination_directory_path, stride_size=64,
                               model_weights_path=os.path.join(this_dir, "saved_model", "spotlight.pth"), patch_size=256):
    """Despeckle specified area in coSAR images using trained MERLIN (spotlight or stripmap weights)

    Args:
        sar_images_path (str): path of sar images
        coordinates_dict (dict): dictionary containing pixel boundaries of the area to despeckle (x_start, x_end, y_start, y_end)
        destination_directory_path (str): path of folder in which results will be stored
        stride_size (int): stride size. Defaults to 64.
        model_weights_path (str): path to model weights (pth file). Defaults to os.path.join(this_dir, "saved_model", "spotlight.pth").
        patch_size (int): patch size. Defaults to 256.

    Returns:
       denoised_image (npy): denoised specified area in image stored in a numpy array (last image contained in sar_images_path)
    """

    logging.info(
        f"""Despeckling images from coordinates using {model_weights_path.split("/")[-1]} weights""")

    processed_images_path = create_empty_folder_in_directory(destination_directory_path=destination_directory_path,
                                                             folder_name="processed_images")
    preprocess_and_store_sar_images_from_coordinates(sar_images_path=sar_images_path, processed_images_path=processed_images_path,
                                                     coordinates_dict=coordinates_dict)

    logging.info(
        f"Starting inference.. Working directory: {os.getcwd()}. Collecting data from {sar_images_path} and storing test results in {destination_directory_path}")

    processed_images_paths = glob((processed_images_path + '/*.npy'))

    denoised_image = Denoiser().denoise_images(images_to_denoise_paths=processed_images_paths, weights_path=model_weights_path, save_dir=destination_directory_path,
                                               stride=stride_size, patch_size=patch_size)

    return denoised_image


def despeckle_from_crop(sar_images_path, destination_directory_path, stride_size=64,
                        model_weights_path=os.path.join(
                            this_dir, "saved_model", "spotlight.pth"),
                        patch_size=256, fixed=True):
    """Despeckle specified area with an integrated cropping tool (made with OpenCV) in coSAR images using trained MERLIN (spotlight or stripmap weights)

    Args:
        sar_images_path (str): path of sar images
        destination_directory_path (str): path of folder in which results will be stored
        stride_size (int): stride size. Defaults to 64.
        model_weights_path (str): path to model weights (pth file). Defaults to os.path.join(this_dir, "saved_model", "spotlight.pth").
        patch_size (int): patch size. Defaults to 256.
        fixed (bool) : If True, crop size is limited to 256*256. Defaults to True

    Returns:
       denoised_image (npy): denoised specified area in image stored in a numpy array (last image contained in sar_images_path)
    """

    logging.info(
        f"""Despeckling cropped images using {model_weights_path.split("/")[-1]} weights""")

    processed_images_path = create_empty_folder_in_directory(destination_directory_path=destination_directory_path,
                                                             folder_name="processed_images")

    images_paths = glob(os.path.join(sar_images_path, "*.cos")) + \
        glob(os.path.join(sar_images_path, "*.npy"))
    for image_path in images_paths:
        # FROM IMAGE PATH RETRIEVE PNG, NPY, REAL , IMAG, THRESHOLD, FILENAME
        image_png, image_data_real, image_data_imag, threshold = get_info_image(
            image_path, destination_directory_path)

        # CROPPING OUR PNG AND REFLECT THE CROP ON REAL AND IMAG
        if fixed:
            print('Fixed mode selected')
            crop_fixed(image_png, image_data_real, image_data_imag,
                       destination_directory_path, processed_images_path)
        else:
            print('Free mode selected')
            crop(image_png, image_data_real, image_data_imag,
                 destination_directory_path, processed_images_path)

        image_data_real_cropped = np.load(
            processed_images_path + '/image_data_real_cropped.npy')
        store_data_and_plot(image_data_real_cropped, threshold,
                            processed_images_path + '/image_data_real_cropped.npy')
        image_data_imag_cropped = np.load(
            processed_images_path + '/image_data_imag_cropped.npy')
        store_data_and_plot(image_data_imag_cropped, threshold,
                            processed_images_path + '/image_data_imag_cropped.npy')

        image_data_real_cropped = image_data_real_cropped.reshape(image_data_real_cropped.shape[0],
                                                                  image_data_real_cropped.shape[1], 1)
        image_data_imag_cropped = image_data_imag_cropped.reshape(image_data_imag_cropped.shape[0],
                                                                  image_data_imag_cropped.shape[1], 1)

        p = Path(image_path)
        np.save(processed_images_path + '/' + p.stem + '_cropped_to_denoise',
                np.concatenate((image_data_real_cropped, image_data_imag_cropped), axis=2))

    processed_cropped_images_paths = glob(
        (processed_images_path + '/*_cropped_to_denoise.npy'))

    logging.info(
        f"Starting inference.. Working directory: {os.getcwd()}. Collecting data from {sar_images_path} and storing test results in {destination_directory_path}")
    denoised_image = Denoiser().denoise_images(images_to_denoise_paths=processed_cropped_images_paths, weights_path=model_weights_path, save_dir=destination_directory_path,
                                               stride=stride_size, patch_size=patch_size)

    return denoised_image
