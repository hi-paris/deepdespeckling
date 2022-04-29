import numpy as np
import torch
from merlinsar.train.utils import *



class Dataset(torch.utils.data.Dataset):
    'characterizes a dataset for pytorch'
    def __init__(self, patche):
        self.patches = patche
      
        # self.patches_MR = patches_MR
        # self.patches_CT = patches_CT

    def __len__(self):
        'denotes the total number of samples'
        return len(self.patches)

    def __getitem__(self, index):
        'Generates one sample of data'
        #select sample
        batch_real = (self.patches[index,:, :, 0])
        batch_imag = (self.patches[index,:, :, 1])
        
        x = torch.tensor(batch_real)
        y = torch.tensor(batch_imag)

        x = x.unsqueeze(0)
        y = y.unsqueeze(0)

        return x, y


class ValDataset(torch.utils.data.Dataset):
    'characterizes a dataset for pytorch'
    def __init__(self, test_set):
        self.files = glob(test_set+'/*.npy')
      
        # self.patches_MR = patches_MR
        # self.patches_CT = patches_CT

    def __len__(self):
        'denotes the total number of samples'
        return len(self.files)

    def __getitem__(self, index):
        'Generates one sample of data'
        #select sample
        eval_data = load_sar_images(self.files)

        current_test=eval_data[index]

        current_test[0,:,:,:] = symetrisation_patch(current_test[0,:,:,:])
        image_real_part = (current_test[:, :, :, 0]).reshape(current_test.shape[0], current_test.shape[1],
                                                              current_test.shape[2], 1)
        image_imag_part = (current_test[:, :, :, 1]).reshape(current_test.shape[0], current_test.shape[1],
                                                              current_test.shape[2], 1)

        return torch.tensor(image_real_part).type(torch.float) , torch.tensor(image_imag_part).type(torch.float) 
