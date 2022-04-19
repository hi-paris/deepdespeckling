import numpy as np
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

from patchify import patchify, unpatchify

image_path="C:/Users/ykemiche/OneDrive - Capgemini/Desktop/fork/merlin/IMAGE_HH_SRA_spot_068.cos"
image_data = cos2mat(image_path) 
np.save('C:/Users/ykemiche/OneDrive - Capgemini/Desktop/fork/merlin/merlin/test_data/test_image_data.npy',image_data)
image = load_sar_images("C:/Users/ykemiche/OneDrive - Capgemini/Desktop/fork/merlin/merlin/test_data/test_image_data.npy").astype(np.float32)


patches = patchify(image, (1,256,256,2), step=64) # split image into 2*3 small 2*2 patches.
patches =patches.reshape(patches.shape[1], patches.shape[2],patches.shape[5],patches.shape[6],patches.shape[7])
patches=patches.reshape(patches.shape[0]*patches.shape[1],patches.shape[2],patches.shape[3],patches.shape[4])

##############################################################################################
for patch in tqdm(patches):

    i_real_part = (patches[:, :, 0]).reshape(patches.shape[0], patches.shape[1],
                patches.shape[2], 1)
    i_imag_part = (patches[:, :, 1]).reshape(patches.shape[0], patches.shape[1],
                patches.shape[2], 1)

    real_to_denoise =torch.tensor(i_real_part)  
    imag_to_denoise=torch.tensor(i_imag_part)

    real_to_denoise = real_to_denoise.type(torch.float32)
    imag_to_denoise = imag_to_denoise.type(torch.float32)

    model = AE(12,1,torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    tmp_clean_image_real = model.forward(real_to_denoise,1).detach().numpy()
    tmp_clean_image_imag = model.forward(imag_to_denoise,1).detach().numpy()


# for patch in patches[]
# assert patches.shape == (2, 3, 2, 2)
# reconstructed_image = unpatchify(patches, image.shape)

# assert (reconstructed_image == image).all()