import torch

from model import *
from utils import *
from model_test import * 
import argparse
import os
from glob import glob

import numpy as np
from load_cosar import cos2mat

M = 10.089038980848645
m = -1.429329123112601



def despeckle(image_path,model,denoiser,destination_directory,stride_size=64,
                model_weights_path="C:/Users/ykemiche/OneDrive - Capgemini/Desktop/fork/merlin/merlin/test/saved_model/model.pth"):
    """ Description
            ----------
            Runs a test instance by calling the test function defined in model.py on a few samples

            Parameters
            ----------
            denoiser : an object

            Returns
            ----------

    """

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
        

        
if __name__ == '__main__':

   #################### INPUTS ##################   

    image_path="C:/Users/ykemiche/OneDrive - Capgemini/Desktop/fork/merlin/merlin/test/IMAGE_HH_SRA_spot_068.cos"
    destination_directory="C:/Users/ykemiche/OneDrive - Capgemini/Desktop/fork/merlin/merlin/test/results"
    # image_part= [1200:1200+256,:256,:]

    # Use argument just if you want to use your own weights
    model_weights_path=""

   #################### INPUTS ##################   

    model = AE(12,1,torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    denoiser=denoiser()
    despeckle(image_path,model,denoiser,destination_directory)

