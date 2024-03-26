import logging
import os
from glob import glob
from deepdespeckling.denoiser import Denoiser

from deepdespeckling.merlin.merlin_denoiser import MerlinDenoiser
from deepdespeckling.sar2sar.sar2sar_denoiser import Sar2SarDenoiser
from deepdespeckling.utils.constants import PATCH_SIZE, STRIDE_SIZE
from deepdespeckling.utils.utils import (crop_image, get_cropping_coordinates, load_sar_image, preprocess_and_store_sar_images_from_coordinates,
                                         create_empty_folder_in_directory, preprocess_and_store_sar_images)


logging.basicConfig(level=logging.INFO)


def get_denoiser(model_name: str, symetrise: bool = True) -> Denoiser:
    """Get the right denoiser object from the model name

    Args:
        model_name (str): model name to be use for despeckling 
        symetrise (bool) : if using spotlight or stripmap model, if True, will symetrise the real and 
            imaginary parts of the noisy image. Defaults to True

    Returns:
        denoiser (Denoiser): the right denoiser, Sar2SarDenoiser or MerlinDenoiser
    """
    if model_name in ["spotlight", "stripmap"]:
        denoiser = MerlinDenoiser(model_name=model_name, symetrise=symetrise)
    elif model_name == "sar2sar":
        denoiser = Sar2SarDenoiser()
    else:
        raise ValueError("The model name doesn't refer to an existing model ")

    return denoiser


def despeckle(sar_images_path: str, destination_directory_path: str, model_name: str = "spotlight",
              patch_size: int = PATCH_SIZE, stride_size: int = STRIDE_SIZE, symetrise: bool = True):
    """Despeckle coSAR images using trained MERLIN (spotlight or stripmap weights) or SAR2SAR

    Args:
        sar_images_path (str): path of sar images
        destination_directory_path (str): path of folder in which results will be stored
        model_name (str): model name, either "spotlight" or "stripmap" to select MERLIN model on the 
            right cosar image format or "sar2sar" for SAR2SAR model. Default to "spotlight"
        patch_size (int): patch size. Defaults to constant PATCH_SIZE.
        stride_size (int): stride size. Defaults to constant STRIDE_SIZE.
        symetrise (bool) : if using spotlight or stripmap model, if True, will symetrise the real and 
            imaginary parts of the noisy image. Defaults to True
    """

    logging.info(
        f"""Despeckling entire images using {model_name} weights""")

    processed_images_path = create_empty_folder_in_directory(destination_directory_path=destination_directory_path,
                                                             folder_name="processed_images")
    preprocess_and_store_sar_images(
        sar_images_path=sar_images_path, processed_images_path=processed_images_path, model_name=model_name)

    logging.info(
        f"Starting inference.. Collecting data from {sar_images_path} and storing test results in {destination_directory_path}")

    denoiser = get_denoiser(model_name=model_name, symetrise=symetrise)
    denoiser.denoise_images(images_to_denoise_path=processed_images_path, save_dir=destination_directory_path,
                            patch_size=patch_size, stride_size=stride_size)


def despeckle_from_coordinates(sar_images_path: str, coordinates_dict: dict, destination_directory_path: str, model_name: str = "spotlight",
                               patch_size: int = PATCH_SIZE, stride_size: int = STRIDE_SIZE, symetrise: bool = True):
    """Despeckle specified area with coordinates in coSAR images using trained MERLIN (spotlight or stripmap weights)

    Args:
        sar_images_path (str): path of sar images
        coordinates_dict (dict): dictionary containing pixel boundaries of the area to despeckle (x_start, x_end, y_start, y_end)
        destination_directory_path (str): path of folder in which results will be stored
        model_name (str): model name, either "spotlight" or "stripmap" to select MERLIN model on the 
            right cosar image format or "sar2sar" for SAR2SAR model. Default to "spotlight"
        patch_size (int): patch size. Defaults to constant PATCH_SIZE.
        stride_size (int): stride size. Defaults to constant STRIDE_SIZE.
        symetrise (bool) : if using spotlight or stripmap model, if True, will symetrise the real and 
            imaginary parts of the noisy image. Defaults to True
    """

    logging.info(
        f"""Despeckling images from coordinates using {model_name} weights""")

    processed_images_path = create_empty_folder_in_directory(destination_directory_path=destination_directory_path,
                                                             folder_name="processed_images")
    preprocess_and_store_sar_images_from_coordinates(sar_images_path=sar_images_path, processed_images_path=processed_images_path,
                                                     coordinates_dict=coordinates_dict, model_name=model_name)

    logging.info(
        f"Starting inference.. Collecting data from {sar_images_path} and storing test results in {destination_directory_path}")

    denoiser = get_denoiser(model_name=model_name, symetrise=symetrise)
    denoiser.denoise_images(images_to_denoise_path=processed_images_path, save_dir=destination_directory_path,
                            patch_size=patch_size, stride_size=stride_size)


def despeckle_from_crop(sar_images_path: str, destination_directory_path: str, model_name: str = "spotlight",
                        patch_size: int = PATCH_SIZE, stride_size: int = STRIDE_SIZE, fixed: bool = True, symetrise: bool = True):
    """Despeckle specified area with an integrated cropping tool (made with OpenCV) in coSAR images using trained MERLIN (spotlight or stripmap weights)

    Args:
        sar_images_path (str): path of sar images
        destination_directory_path (str): path of folder in which results will be stored
        patch_size (int): patch size. Defaults to constant PATCH_SIZE.
        stride_size (int): stride size. Defaults to constant STRIDE_SIZE.
        model_name (str): model name, either "spotlight" or "stripmap" to select MERLIN model on the 
            right cosar image format or "sar2sar" for SAR2SAR model. Default to "spotlight"
        fixed (bool) : If True, crop size is limited to 256*256. Defaults to True
        symetrise (bool) : if using spotlight or stripmap model, if True, will symetrise the real and 
            imaginary parts of the noisy image. Defaults to True
    """

    logging.info(
        f"""Cropping and despeckling images using {model_name} weights""")

    processed_images_path = create_empty_folder_in_directory(destination_directory_path=destination_directory_path,
                                                             folder_name="processed_images")

    ext = "cos" if model_name in ["spotlight", "stripmap"] else "tiff"
    images_paths = glob(os.path.join(sar_images_path, f"*.{ext}")) + \
        glob(os.path.join(sar_images_path, "*.npy"))

    for i, image_path in enumerate(images_paths):
        # Load image for cropping
        image = load_sar_image(image_path)

        # Get cropping coordinates from the first image of the list of images to crop and despeckle
        if i == 0:
            cropping_coordinates = get_cropping_coordinates(
                image=image, fixed=fixed, destination_directory_path=destination_directory_path, model_name=model_name)

        # Crop image using stored cropping coordinates and store it in processed_images_path
        crop_image(image, image_path, cropping_coordinates, model_name,
                   processed_images_path)

    logging.info(
        f"Starting inference.. Collecting data from {sar_images_path} and storing results in {destination_directory_path}")

    denoiser = get_denoiser(model_name=model_name, symetrise=symetrise)
    denoiser.denoise_images(images_to_denoise_path=processed_images_path, save_dir=destination_directory_path,
                            patch_size=patch_size, stride_size=stride_size)
