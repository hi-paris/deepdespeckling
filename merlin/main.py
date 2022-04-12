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

parser = argparse.ArgumentParser(description='')

parser.add_argument('--test_dir', dest='test_dir', default='C:/Users/ykemiche/OneDrive - Capgemini/Desktop/merlin/merlin_package/merlin/merlin/results', help='test examples are saved here')

parser.add_argument('--model_path', dest='model_path', default='C:/Users/ykemiche/OneDrive - Capgemini/Desktop/merlin/merlin_package/merlin/merlin/saved_model/model.pth', help='saved model weights')

parser.add_argument('--test_data', dest='test_data', default='C:/Users/ykemiche/OneDrive - Capgemini/Desktop/merlin/merlin_package/merlin/merlin/test_data',
                    help='data set for testing')
parser.add_argument('--stride_size', dest='stride_size', type=int, default=64,
                    help='define stride when image dim exceeds 264')
                    
args = parser.parse_args()


def denoiser_test(model,denoiser):
    """ Description
            ----------
            Runs a test instance by calling the test function defined in model.py on a few samples

            Parameters
            ----------
            denoiser : an object

            Returns
            ----------

    """
    test_data = args.test_data

    print(
        "[*] Start testing on real data. Working directory: %s. Collecting data from %s and storing test results in %s" % (
            os.getcwd(), test_data, args.test_dir))
            
    test_files = glob((test_data + '/*.npy'))
    print(test_files)
    denoiser.test(model,test_files,args.model_path, save_dir=args.test_dir, dataset_dir=test_data,
                  stride=args.stride_size)
    

if __name__ == '__main__':

    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    model = AE(12,1,torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    denoiser=denoiser()
    denoiser_test(model,denoiser)


