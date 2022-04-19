
import torch

from model import *
from utils import *
from model_test import * 
import argparse
import os
from glob import glob
from tqdm import tqdm

import numpy as np
from load_cosar import cos2mat


# image_path="C:/Users/ykemiche/OneDrive - Capgemini/Desktop/fork/merlin/IMAGE_HH_SRA_spot_068.cos"

# image_data = cos2mat(image_path) 
# print(image_data.shape)

# real=image_data[1200:1200+256,:256,:]

for i in tqdm(range(15531)):
    real_image = load_sar_images("C:/Users/ykemiche/OneDrive - Capgemini/Desktop/fork/merlin/merlin/test_data/test_image_data.npy").astype(np.float32)

    i_real_part = (real_image[:, :, :, 0]).reshape(real_image.shape[0], real_image.shape[1],
                        real_image.shape[2], 1)
    i_imag_part = (real_image[:, :, :, 1]).reshape(real_image.shape[0], real_image.shape[1],
                        real_image.shape[2], 1)

    real_to_denoise =torch.tensor(i_real_part)  
    imag_to_denoise=torch.tensor(i_imag_part)

    real_to_denoise = real_to_denoise.type(torch.float32)
    imag_to_denoise = imag_to_denoise.type(torch.float32)

    model = AE(12,1,torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    tmp_clean_image_real = model.forward(real_to_denoise,1).detach().numpy()
    tmp_clean_image_imag = model.forward(real_to_denoise,1).detach().numpy()

