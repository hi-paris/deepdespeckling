import torch
import numpy as np


class Model(torch.nn.Module):

    def __init__(self, device: str, height: int, width: int):
        super().__init__()

        self.device = device

        self.height = height
        self.width = width

        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.leaky = torch.nn.LeakyReLU(0.1)

        self.enc0 = torch.nn.Conv2d(in_channels=1, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same', device=self.device)
        self.enc1 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same', device=self.device)
        self.enc2 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same', device=self.device)
        self.enc3 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same', device=self.device)
        self.enc4 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same', device=self.device)
        self.enc5 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same', device=self.device)
        self.enc6 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same', device=self.device)

        self.dec5 = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same', device=self.device)
        self.dec5b = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                     padding='same', device=self.device)
        self.dec4 = torch.nn.Conv2d(in_channels=144, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same', device=self.device)
        self.dec4b = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                     padding='same', device=self.device)
        self.dec3 = torch.nn.Conv2d(in_channels=144, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same', device=self.device)
        self.dec3b = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                     padding='same', device=self.device)
        self.dec2 = torch.nn.Conv2d(in_channels=144, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same', device=self.device)
        self.dec2b = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                     padding='same', device=self.device)
        self.dec1a = torch.nn.Conv2d(in_channels=97, out_channels=64, kernel_size=(3, 3), stride=(1, 1),
                                     padding='same', device=self.device)
        self.dec1b = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1),
                                     padding='same', device=self.device)
        self.dec1 = torch.nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same', device=self.device)

        self.upscale2d = torch.nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x: np.array) -> np.array:
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
        x = torch.reshape(x, [1, 1, self.height, self.width])
        # x = torch.permute(x, (0, 3, 1, 2))
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
