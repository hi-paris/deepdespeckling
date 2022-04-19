#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from glob import glob
import os

import numpy as np
import torch

from Dataset import *
from model import *
from utils import *


torch.manual_seed(2)


def evaluate(model, loader):
    outputs = [model.validation_step(batch) for batch in loader]
    outputs = torch.tensor(outputs).T
    loss, accuracy = torch.mean(outputs, dim=1)
    return {"loss": loss.item(), "accuracy": accuracy.item()}


def save_model(model,destination_folder):
    """
      save the ".pth" model in destination_folder
    """
    # Check whether the specified path exists or not
    isExist = os.path.exists(destination_folder)

    if not isExist:

      # Create a new directory because it does not exist
      os.makedirs(destination_folder)
      print("The new directory is created!")

      torch.save(model.state_dict(),destination_folder+"/model.pth")

    else:
      torch.save(model.state_dict(),destination_folder+"/model.pth")


def fit(model,train_loader,val_loader,epochs,lr_list,eval_files,eval_set,sample_dir):
  """ Fit the model according to the given evaluation data and parameters.

  Parameters
  ----------
  model : model as defined in main
  train_loader : Pytorch's DataLoader of training data
  val_loader : Pytorch's DataLoader of validation data
  lr_list : list of learning rates
  eval_files : .npy files used for evaluation in training
  eval_set : directory of dataset used for evaluation in training

  Returns
  ----------
  self : object
    Fitted estimator.

  """

  train_losses = []
  val_losses=[]
  history={}
  epoch_num=0
  for epoch in range(epochs):
      epoch_num=epoch_num+1
      print("\nEpoch", epoch+1)
      print("***************** \n")
      optimizer = torch.optim.Adam(model.parameters(), lr=lr_list[epoch])

      #Train
      for i, batch in enumerate(train_loader, 0):

            optimizer.zero_grad()
            loss = model.training_step(batch,i)
            train_losses.append(loss)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            #running_loss += loss.item()     # extract the loss value
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss))
            # zero the loss


          # #Validate
      with torch.no_grad():
        image_num=0
        for batch in val_loader:
            val_loss=model.validation_step(batch,image_num,epoch_num,eval_files,eval_set,sample_dir)
            image_num=image_num+1

  history["train_loss"]=train_losses

  return history


def create_model(batch_size=12,val_batch_size=1,device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),from_pretrained=False):
  """ Runs the denoiser algorithm for the training and evaluation dataset


  """
  weights_path="C:/Users/ykemiche/OneDrive - Capgemini/Desktop/fork/merlin/merlin/train/saved_model/model.pth"

  model = AE(batch_size,val_batch_size,device)
  model.to(device)
  if from_pretrained == True:
        
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))


  return model

def denoiser_train(model,lr_list,nb_epoch,training_set_directory,validation_set_directory,sample_directory,save_directory,patch_size=256,batch_size=12,val_batch_size=1,stride_size=128,n_data_augmentation=1):
  """ Runs the denoiser algorithm for the training and evaluation dataset

  Parameters
  ----------
  model : model as defined in main
  lr_list : list of learning rates

  Returns
  ----------
  history : list of both training and validation loss

  """
  # Prepare train DataLoader
  train_data = load_train_data(training_set_directory,patch_size,batch_size,stride_size,n_data_augmentation) # range [0; 1]
  train_dataset = Dataset(train_data)
  train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=True)

  # Prepare Validation DataLoader
  eval_dataset = ValDataset(validation_set_directory)
  eval_loader = torch.utils.data.DataLoader(eval_dataset,batch_size=val_batch_size,shuffle=False,drop_last=True)

  eval_files = glob(validation_set_directory+'/*.npy')
  # Train the model
  history =fit(model,train_loader,eval_loader,nb_epoch,lr_list,eval_files,validation_set_directory,sample_directory)

  # Save the model
  save_model(model,save_directory)
  print("\n model saved at :",save_directory)
   
  return history

def main():
      
  #################### INPUTS ##################   
  nb_epoch=1

  lr = 0.001 * np.ones([nb_epoch])
  lr[6:20] = lr[0]/10
  lr[20:] = lr[0]/100

  training_set_directory="C:/Users/ykemiche/OneDrive - Capgemini/Desktop/fork/merlin/merlin/train/data/training"
  validation_set_directory="C:/Users/ykemiche/OneDrive - Capgemini/Desktop/fork/merlin/merlin/train/data/test"
  save_directory="C:/Users/ykemiche/OneDrive - Capgemini/Desktop/fork/merlin/merlin/train/saved_model"
  sample_directory="C:/Users/ykemiche/OneDrive - Capgemini/Desktop/fork/merlin/merlin/train/data/sample"
  from_pretrained=False
  ###########################################

  model = create_model(from_pretrained=from_pretrained)
  denoiser_train(model,lr,nb_epoch,training_set_directory,validation_set_directory,sample_directory,save_directory)



if __name__ == '__main__':
    main()
