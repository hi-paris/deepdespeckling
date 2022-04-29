import torch

from merlinsar.test.model import *
from merlinsar.test.utils import *
from merlinsar.test.model_test import * 
import argparse
import os
from glob import glob

import numpy as np
from merlinsar.test.load_cosar import cos2mat

M = 10.089038980848645
m = -1.429329123112601



def despeckle(image_path,destination_directory,stride_size=64,
                model_weights_path="merlin-sar/merlin/test/saved_model/model.pth"):
    """ Description
            ----------
            Runs a test instance by calling the test function defined in model.py on a few samples

            Parameters
            ----------
            denoiser : an object

            Returns
            ----------

    """
    model = AE(12,1,torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    denoiser=Denoiser()

    if not os.path.exists(destination_directory+'/processed_image'):
        os.mkdir(destination_directory+'/processed_image')
    
    test_data=destination_directory+'/processed_image'
    image_data = cos2mat(image_path) 

    np.save(test_data+'/test_image_data.npy',image_data[1200:1200+256,:256,:])

    print(
        "[*] Start testing on real data. Working directory: %s. Collecting data from %s and storing test results in %s" % (
            os.getcwd(), destination_directory, destination_directory))
            
    test_files = glob((test_data + '/*.npy'))
    print(test_files)
    denoiser.test(model,test_files,model_weights_path, save_dir=destination_directory, dataset_dir=destination_directory,
                  stride=stride_size)
        

def despeckle_from_crop(image_path,destination_directory,stride_size=64,
                model_weights_path="merlin-sar/merlin/test/saved_model/model.pth"):
    """ Description
            ----------
            Runs a test instance by calling the test function defined in model.py on a few samples

            Parameters
            ----------
            denoiser : an object

            Returns
            ----------

    """
    model = AE(12,1,torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    denoiser=Denoiser()

    if not os.path.exists(destination_directory+'\\processed_image'):
        os.mkdir(destination_directory+'\\processed_image')

    test_data=destination_directory+'\\processed_image'

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
    denoiser.test(model,test_files,model_weights_path, save_dir=destination_directory, dataset_dir=destination_directory,
                  stride=stride_size)
