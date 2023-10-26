import numpy as np

from merlinsar.train.utils import *
from scipy import special


# DEFINE PARAMETERS OF SPECKLE AND NORMALIZATION FACTOR
M = 10.089038980848645
m = -1.429329123112601
L = 1
c = (1 / 2) * (special.psi(L) - np.log(L))
cn = c / (M - m)  # normalized (0,1) mean of log speckle

import torch
import numpy as np


# Nearest-neighbor up-scaling layer.
'''def upscale2d(x, factor=2):
    """ Description
            ----------
            Run a nearest-neighbor up-scaling layer for a given factor on an object (image) x

            This is a common quality-enhancing algorithm  that doubles the dimensions of the input.

            Every pixel in low-res is thereafter transformed into 4 pixels.

            Parameters
            ----------
            x : an image object

            factor : figure by how many times ou want to multiply the input's initial dimensions

            Returns
            ----------
            x if factor is 1
            or
            a reshaped x if factor is different from 1
        """
    assert isinstance(factor, int) and factor >= 1
    if factor == 1:
        return x
    # with tf.compat.v1.variable_scope('Upscale2D'):
    s = x.shape
    x = torch.reshape(x, [-1, s[1], 1, s[2], 1, s[3]])
    x = torch.tile(x, [1, 1, factor, 1, factor, 1])
    x = torch.reshape(x, [-1, s[1] , s[2] * factor, s[3]* factor])
    
    
    s = x.shape
    x = torch.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
    x = torch.tile(x, [1, 1, 1, factor, 1, factor])
    x = torch.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])

    return x'''



class Model(torch.nn.Module):

    def __init__(self,height,width,batch_size,eval_batch_size,device):
        super().__init__()

        self.batch_size=batch_size
        self.eval_batch_size=eval_batch_size
        self.device=device

        self.height = height
        self.width = width

        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.leaky = torch.nn.LeakyReLU(0.1)

        self.enc0 = torch.nn.Conv2d(in_channels=1, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.enc1 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.enc2 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.enc3 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.enc4 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.enc5 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.enc6 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')

        self.dec5 = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.dec5b = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                     padding='same')
        self.dec4 = torch.nn.Conv2d(in_channels=144, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.dec4b = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                     padding='same')
        self.dec3 = torch.nn.Conv2d(in_channels=144, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.dec3b = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                     padding='same')
        self.dec2 = torch.nn.Conv2d(in_channels=144, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.dec2b = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                     padding='same')
        self.dec1a = torch.nn.Conv2d(in_channels=97, out_channels=64, kernel_size=(3, 3), stride=(1, 1),
                                     padding='same')
        self.dec1b = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1),
                                     padding='same')
        self.dec1 = torch.nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')

        self.upscale2d = torch.nn.UpsamplingNearest2d(scale_factor=2)


    def forward(self, x ,batch_size):
        """  Defines a class for an autoencoder algorithm for an object (image) x

        An autoencoder is a specific type of feedforward neural networks where the
        input is the same as the
        output. It compresses the input into a lower-dimensional code and then 
        reconstruct the output from this representattion. It is a dimensionality 
        reduction algorithm

        Parameters
        ----------
        x : np.array
        a numpy array containing image 

        Returns
        ----------
        x-n : np.array
        a numpy array containing the denoised image i.e the image itself minus the noise

        """
      
        x=torch.reshape(x, [batch_size, 1, self.height, self.width])
        skips = [x]

        n = x

        # ENCODER
        n = self.leaky(self.enc0(n))
        n = self.leaky(self.enc1(n))
        n = self.pool(n)
        skips.append(n)

        n = self.leaky(self.enc2(n))
        n = self.pool(n)
        skips.append(n)

        n = self.leaky(self.enc3(n))
        n = self.pool(n)
        skips.append(n)

        n = self.leaky(self.enc4(n))
        n = self.pool(n)
        skips.append(n)

        n = self.leaky(self.enc5(n))
        n = self.pool(n)
        n = self.leaky(self.enc6(n))


        # DECODER
        n = self.upscale2d(n)
        n = torch.cat((n, skips.pop()), dim=1)
        n = self.leaky(self.dec5(n))
        n = self.leaky(self.dec5b(n))

        n = self.upscale2d(n)
        n = torch.cat((n, skips.pop()), dim=1)
        n = self.leaky(self.dec4(n))
        n = self.leaky(self.dec4b(n))

        n = self.upscale2d(n)
        n = torch.cat((n, skips.pop()), dim=1)
        n = self.leaky(self.dec3(n))
        n = self.leaky(self.dec3b(n))

        n = self.upscale2d(n)
        n = torch.cat((n, skips.pop()), dim=1)
        n = self.leaky(self.dec2(n))
        n = self.leaky(self.dec2b(n))

        n = self.upscale2d(n)
        n = torch.cat((n, skips.pop()), dim=1)
        n = self.leaky(self.dec1a(n))
        n = self.leaky(self.dec1b(n))

        n = self.dec1(n)

        return x - n

    def loss_function(self,output,target,batch_size):
      """ Defines and runs the loss function
        
      Parameters
      ----------
      output : 
      target :
      batch_size :

      Returns
      ----------
      loss: float
          The value of loss given your output, target and batch_size

      """

      M = 10.089038980848645
      m = -1.429329123112601
 
      # ----- loss -----
      log_hat_R = 2*(output*(M-m)+m)
      hat_R = torch.exp(log_hat_R)+1e-6 # must be nonzero
      b_square = torch.square(target)
      loss = (1/batch_size)*torch.mean( 0.5*log_hat_R+b_square/hat_R  ) #+ tf.losses.get_regularization_loss()
      return loss
 
    def training_step(self, batch,batch_number):
      
      """ Train the model with the training set

      Parameters
      ----------
      batch : a subset of the training date
      batch_number : ID identifying the batch

      Returns
      -------
      loss : float
        The value of loss given the batch
        
      """
      M = 10.089038980848645
      m = -1.429329123112601

      x, y = batch
      x=x.to(self.device)
      y=y.to(self.device)


        

      if (batch_number%2==0):
        x=(torch.log(torch.square(x)+1e-3)-2*m)/(2*(M-m))
        out = self.forward(x,self.batch_size)
        loss = self.loss_function(out, y,self.batch_size)

      else:
        y=(torch.log(torch.square(y)+1e-3)-2*m)/(2*(M-m))
        out = self.forward(y,self.batch_size)
        loss = self.loss_function(out,x,self.batch_size)
          
      return loss

    def validation_step(self, batch,image_num,epoch_num,eval_files,eval_set,sample_dir):
      """ Test the model with the validation set

      Parameters
      ----------
      batch : a subset of data
      image_num : an ID identifying the feeded image
      epoch_num : an ID identifying the epoch
      eval_files : .npy files used for evaluation in training
      eval_set : directory of dataset used for evaluation in training

      Returns
      ----------
      output_clean_image : a np.array
            
      """

      image_real_part,image_imaginary_part = batch

      image_real_part=image_real_part.to(self.device) 
      image_imaginary_part=image_imaginary_part.to(self.device)

      # Normalization
      image_real_part_normalized=(torch.log(torch.square(image_real_part)+1e-3)-2*m)/(2*(M-m))
      image_imaginary_part_normalized=(torch.log(torch.square(image_imaginary_part)+1e-3)-2*m)/(2*(M-m))
            
      out_real = self.forward(image_real_part_normalized,self.eval_batch_size)
      out_imaginary = self.forward(image_imaginary_part_normalized,self.eval_batch_size)

      output_clean_image = 0.5*(np.square(denormalize_sar(out_real.cpu().numpy()))+np.square(denormalize_sar(out_imaginary.cpu().numpy())))
      # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

      noisyimage = np.squeeze(np.sqrt(np.square(image_real_part.cpu().numpy())+np.square(image_imaginary_part.cpu().numpy())))
      outputimage = np.sqrt(np.squeeze(output_clean_image))

      # calculate PSNR
      psnr = cal_psnr(outputimage, noisyimage)
      print("img%d PSNR: %.2f" % (image_num , psnr))

      # rename and save
      imagename = eval_files[image_num].replace(eval_set, "")
      imagename = imagename.replace('.npy', '_epoch_' + str(epoch_num) + '.npy')

      save_sar_images(outputimage, noisyimage, imagename,sample_dir)

      return output_clean_image


    # def optimizer(self):
    #   return torch.optim.Adam