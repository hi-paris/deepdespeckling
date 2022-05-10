import torch

from merlinsar.test.model import *
from merlinsar.test.utils import *
from merlinsar.test.model_test import * 
import os
from glob import glob

import numpy as np
from merlinsar.test.load_cosar import cos2mat

M = 10.089038980848645
m = -1.429329123112601

this_dir, this_filename = os.path.split(__file__)


def despeckle(image_path,destination_directory,stride_size=64,
                model_weights_path= os.path.join(this_dir, "saved_model", "model.pth"),patch_size=256,height=256,width=256):

    """ Description
            ----------
            Runs a test instance by calling the test function defined in model.py on a few samples

            Parameters
            ----------
            denoiser : an object

            Returns
            ----------

    """

    denoiser=Denoiser()

    if not os.path.exists(destination_directory+'/processed_image'):
        os.mkdir(destination_directory+'/processed_image')
    
    test_data=destination_directory+'/processed_image'

    filelist = glob(os.path.join(test_data, "*"))
    for f in filelist:
        os.remove(f)

    image_data = cos2mat(image_path) 

    np.save(test_data+'/test_image_data.npy',image_data)

    print(
        "[*] Start testing on real data. Working directory: %s. Collecting data from %s and storing test results in %s" % (
            os.getcwd(), destination_directory, destination_directory))
            
    test_files = glob((test_data + '/*.npy'))
    print(test_files)

    denoiser.test(test_files,model_weights_path, save_dir=destination_directory,
                  stride=stride_size,patch_size=patch_size,height=height,width=width)
        

def despeckle_from_coordinates(image_path,coordinates_dict,destination_directory,stride_size=64,
                model_weights_path= os.path.join(this_dir, "saved_model", "model.pth"),patch_size=256,height=256,width=256):
    """ Description
            ----------
            Runs a test instance by calling the test function defined in model.py on a few samples

            Parameters
            ----------
            denoiser : an object

            Returns
            ----------

    """

    x_start=coordinates_dict["x_start"]
    x_end=coordinates_dict["x_end"]
    y_start=coordinates_dict["y_start"]
    y_end=coordinates_dict["y_end"]

    denoiser=Denoiser()

    if not os.path.exists(destination_directory+'/processed_image'):
        os.mkdir(destination_directory+'/processed_image')
    
    test_data=destination_directory+'/processed_image'
 
    filelist = glob(os.path.join(test_data, "*"))
    for f in filelist:
        os.remove(f)

    image_data = cos2mat(image_path) 

    np.save(test_data+'/test_image_data.npy',image_data[x_start:x_end,y_start:y_end,:])

    print(
        "[*] Start testing on real data. Working directory: %s. Collecting data from %s and storing test results in %s" % (
            os.getcwd(), destination_directory, destination_directory))
            
    test_files = glob((test_data + '/*.npy'))
    print(test_files)
    denoiser.test(test_files,model_weights_path, save_dir=destination_directory,
                  stride=stride_size,patch_size=patch_size,height=height,width=width)



def despeckle_from_crop(image_path,destination_directory,stride_size=64,
                model_weights_path= os.path.join(this_dir, "saved_model", "model.pth"),patch_size=256,height=256,width=256):
    """ Description
            ----------
            Runs a test instance by calling the test function defined in model.py on a few samples

            Parameters
            ----------
            denoiser : an object

            Returns
            ----------

    """

    denoiser=Denoiser()

    if not os.path.exists(destination_directory+'\\processed_image'):
        os.mkdir(destination_directory+'\\processed_image')

    test_data=destination_directory+'\\processed_image'

    filelist = glob(os.path.join(test_data, "*"))
    for f in filelist:
        os.remove(f)


    # FROM IMAGE PATH RETRIEVE PNG, NPY, REAL , IMAG, THRESHOLD, FILENAME
    image_png , image_data, image_data_real , image_data_imag, threshold, filename = get_info_image(image_path,destination_directory)

    # CROPPING OUR PNG AND REFLECT THE CROP ON REAL AND IMAG
    cropping = False
    crop(image_png,image_data_real, image_data_imag, destination_directory,test_data,cropping)


    image_data_real_cropped = np.load(test_data+'\\image_data_real_cropped.npy')
    store_data_and_plot(image_data_real_cropped, threshold, test_data+'\\image_data_real_cropped.npy')
    image_data_imag_cropped =np.load(test_data+'\\image_data_imag_cropped.npy')
    store_data_and_plot(image_data_imag_cropped, threshold, test_data+'\\image_data_imag_cropped.npy')

    image_data_real_cropped= image_data_real_cropped.reshape(image_data_real_cropped.shape[0],image_data_real_cropped.shape[1],1)
    image_data_imag_cropped = image_data_imag_cropped.reshape(image_data_imag_cropped.shape[0],
                                                              image_data_imag_cropped.shape[1], 1)

    np.save(test_data + '/test_image_data_cropped.npy', np.concatenate((image_data_real_cropped, image_data_imag_cropped), axis=2))

    print(
        "[*] Start testing on real data. Working directory: %s. Collecting data from %s and storing test results in %s" % (
            os.getcwd(), destination_directory, destination_directory))

    test_files = glob((test_data +'/test_image_data_cropped.npy'))
    print(test_files)
    denoiser.test(test_files,model_weights_path, save_dir=destination_directory,
                  stride=stride_size,patch_size=patch_size,height=height,width=width)
