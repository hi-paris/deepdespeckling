import os
from glob import glob
from pathlib import Path
import numpy as np

from deepdespeckling.merlin.test.load_cosar import cos2mat
from deepdespeckling.merlin.test.model_test import Denoiser
from deepdespeckling.utils.utils import crop, crop_fixed, get_info_image, store_data_and_plot


this_dir, this_filename = os.path.split(__file__)


def despeckle_spotlight(image_path, destination_directory, stride_size=64,
                        model_weights_path=os.path.join(this_dir, "saved_model", "spotlight.pth"), patch_size=256):
    """ Description
            ----------
            Runs a test instance by calling the test function defined in model.py on a few samples

            Parameters
            ----------
            denoiser : an object

            Returns
            ----------

    """

    denoiser = Denoiser()

    if not os.path.exists(destination_directory + '/processed_image'):
        os.mkdir(destination_directory + '/processed_image')

    test_data = destination_directory + '/processed_image'

    filelist = glob(os.path.join(test_data, "*"))
    for f in filelist:
        os.remove(f)

    # check the extension of the file
    if Path(image_path).suffix == ".npy":
        image_data = np.load(image_path)
    else:
        image_data = cos2mat(image_path)

    imagename = image_path.split('/')[-1].split('.')[0]
    np.save(test_data + '/' + imagename +'.npy', image_data)

    print(
        "[*] Start testing on real data. Working directory: %s. Collecting data from %s and storing test results in %s" % (
            os.getcwd(), image_path, destination_directory))

    test_files = glob((test_data + '/*.npy'))

    denoiser.test(test_files, model_weights_path, save_dir=destination_directory,
                  stride=stride_size, patch_size=patch_size)


def despeckle_from_coordinates_spotlight(image_path, coordinates_dict, destination_directory, stride_size=64,
                                         model_weights_path=os.path.join(this_dir, "saved_model", "spotlight.pth"),
                                         patch_size=256):
    """ Description
            ----------
            Runs a test instance by calling the test function defined in model.py on a few samples

            Parameters
            ----------
            denoiser : an object

            Returns
            ----------

    """

    x_start = coordinates_dict["x_start"]
    x_end = coordinates_dict["x_end"]
    y_start = coordinates_dict["y_start"]
    y_end = coordinates_dict["y_end"]

    denoiser = Denoiser()

    if not os.path.exists(destination_directory + '/processed_image'):
        os.mkdir(destination_directory + '/processed_image')

    test_data = destination_directory + '/processed_image'

    filelist = glob(os.path.join(test_data, "*"))
    for f in filelist:
        os.remove(f)

    # check the extension of the file
    if Path(image_path).suffix == ".npy":
        image_data = np.load(image_path)
    else:
        image_data = cos2mat(image_path)

    imagename = image_path.split('/')[-1].split('.')[0]
    np.save(test_data + '/' + imagename +'.npy', image_data[x_start:x_end, y_start:y_end, :])

    print(
        "[*] Start testing on real data. Working directory: %s. Collecting data from %s and storing test results in %s" % (
            os.getcwd(), image_path, destination_directory))

    test_files = glob((test_data + '/*.npy'))

    denoiser.test_functions(test_files, model_weights_path, save_dir=destination_directory,
                  stride=stride_size, patch_size=patch_size)


def despeckle_from_crop_spotlight(image_path, destination_directory, stride_size=64,
                                  model_weights_path=os.path.join(this_dir, "saved_model", "spotlight.pth"),
                                  patch_size=256,
                                  fixed=True):
    print('value ofd fixed in despeckle from crop', fixed)
    """ The despeckling function with an integrated cropping tool made with OpenCV.
    The ideal choice if you need to despeckle only a certain area of your high-res image. Results are saved in the
    directory provided in the 'destination directory'
            Parameters
            ----------
            image_path: string
            the path leading to the image to be despceckled
            destination_directory: string
            path leading to the results folder
            stride_size: integer or tuple
            stride of the autoencoder
            model_weights_path: string
            path leading to the weights of our pre-trained model. Value by default is our weights.
            patch_size: integer
            Area size of the sub-image to be processed. Value by default is 256.
            height: integer
            Height of the image. Value by default is 256.
            width: integer
            Width of the image. Value by default is 256.
            fixed: bool
            If True, crop size is limited to 256*256
            Returns
            ----------
            None
    """

    denoiser = Denoiser()

    if not os.path.exists(destination_directory + '\\processed_image'):
        os.mkdir(destination_directory + '\\processed_image')

    test_data = destination_directory + '\\processed_image'

    filelist = glob(os.path.join(test_data, "*"))
    for f in filelist:
        os.remove(f)

    # FROM IMAGE PATH RETRIEVE PNG, NPY, REAL , IMAG, THRESHOLD, FILENAME
    image_png, image_data, image_data_real, image_data_imag, threshold, filename = get_info_image(image_path,
                                                                                                  destination_directory)

    # CROPPING OUR PNG AND REFLECT THE CROP ON REAL AND IMAG
    if fixed:
        print('Fixed mode selected')
        crop_fixed(image_png, image_data_real, image_data_imag, destination_directory, test_data)
    else:
        print('Free mode selected')
        crop(image_png, image_data_real, image_data_imag, destination_directory, test_data)

    image_data_real_cropped = np.load(test_data + '\\image_data_real_cropped.npy')
    store_data_and_plot(image_data_real_cropped, threshold, test_data + '\\image_data_real_cropped.npy')
    image_data_imag_cropped = np.load(test_data + '\\image_data_imag_cropped.npy')
    store_data_and_plot(image_data_imag_cropped, threshold, test_data + '\\image_data_imag_cropped.npy')

    image_data_real_cropped = image_data_real_cropped.reshape(image_data_real_cropped.shape[0],
                                                              image_data_real_cropped.shape[1], 1)
    image_data_imag_cropped = image_data_imag_cropped.reshape(image_data_imag_cropped.shape[0],
                                                              image_data_imag_cropped.shape[1], 1)

    p = Path(image_path)
    np.save(test_data + '/' + p.stem + '_cropped',
            np.concatenate((image_data_real_cropped, image_data_imag_cropped), axis=2))

    print(
        "[*] Start testing on real data. Working directory: %s. Collecting data from %s and storing test results in %s" % (
            os.getcwd(), image_path, destination_directory)+"\n")

    test_files = glob((test_data + '/' + p.stem + '_cropped.npy'))
    denoiser.test(test_files, model_weights_path, save_dir=destination_directory,
                  stride=stride_size, patch_size=patch_size)
