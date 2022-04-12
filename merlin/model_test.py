from utils import *
from model import *
import torch
import numpy as np

M = 10.089038980848645
m = -1.429329123112601

class denoiser(object):
    """ Description
                ----------
                A set of initial conditions, and transformations on the Y

                Parameters
                ----------
                denoiser : an object

                Returns
                ----------
    """

    
    def __init__(self, input_c_dim=1):

        self.input_c_dim = input_c_dim


    def load(self,model,weights_path):
        """ Description
                    ----------
                    Restores a checkpoint located in a checkpoint repository

                    Parameters
                    ----------
                    checkpoint_dir : a path leading to the checkpoint file

                    Returns
                    ----------
                    True : Restoration is a success
                    False: Restoration has failed
        """
        print("[*] Loading the model...")

        model.load_state_dict(torch.load(weights_path))

        return model

     


    def test(self,model, test_files, weights_path, save_dir, dataset_dir, stride):

        """ Description
                    ----------
                    The function that does the job. Should be merged with main.py ?

                    Parameters
                    ----------
                    test_files : a path leading to the checkpoint file
                    ckpt_dir : repository containing the checkpoint (and weights)
                    save_dir : repository to save sar images, real images and noisy images
                    dataset_dir : the path to the test data
                    stride : number of bytes from one row of pixels in memory to the next row of pixels in memory

                    Returns
                    ----------
                    True : Restoration is a success
                    False: Restoration has failed

        """

        """Test MERLIN"""

        #tf.compat.v1.global_variables_initializer().run()
        assert len(test_files) != 0, 'No testing data!'

        loaded_model = AE(12,1,torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        loaded_model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

        # loaded_model = self.load(model,weights_path)
        print()        
        print(" [*] Load weights SUCCESS...")
        print("[*] start testing...")
        
        for idx in range(len(test_files)):
            real_image = load_sar_images(test_files[idx]).astype(np.float32)
            i_real_part = (real_image[:, :, :, 0]).reshape(real_image.shape[0], real_image.shape[1],
                                                           real_image.shape[2], 1)
            i_imag_part = (real_image[:, :, :, 1]).reshape(real_image.shape[0], real_image.shape[1],
                                                           real_image.shape[2], 1)


            # scan on image dimensions
            stride = 64
            pat_size = 256

            # Pad the image
            im_h = np.size(real_image, 1)
            im_w = np.size(real_image, 2)

            count_image = np.zeros(i_real_part.shape)
            output_clean_image_1 = np.zeros(i_real_part.shape)
            output_clean_image_2 = np.zeros(i_real_part.shape)

            if im_h == pat_size:
                x_range = list(np.array([0]))

            else:
                x_range = list(range(0, im_h - pat_size, stride))
                if (x_range[-1] + pat_size) < im_h: x_range.extend(range(im_h - pat_size, im_h - pat_size + 1))

            if im_w == pat_size:
                y_range = list(np.array([0]))

            else:
                y_range = list(range(0, im_w - pat_size, stride))
                if (y_range[-1] + pat_size) < im_w: y_range.extend(range(im_w - pat_size, im_w - pat_size + 1))

            for x in x_range:
                for y in y_range:
                  
                    real_to_denoise, imag_to_denoise = symetrisation_patch_test(i_real_part[:, x:x + pat_size, y:y + pat_size, :],i_imag_part[:, x:x + pat_size, y:y + pat_size, :])

                    real_to_denoise =torch.tensor(real_to_denoise)  
                    imag_to_denoise=torch.tensor(imag_to_denoise)   

                    real_to_denoise = real_to_denoise.type(torch.float32)
                    imag_to_denoise = imag_to_denoise.type(torch.float32)

                    real_to_denoise=(torch.log(torch.square(real_to_denoise)+1e-3)-2*m)/(2*(M-m))
                    imag_to_denoise=(torch.log(torch.square(imag_to_denoise)+1e-3)-2*m)/(2*(M-m))

            
                    tmp_clean_image_real = loaded_model.forward(real_to_denoise,1).detach().numpy()                          
                    tmp_clean_image_real=np.moveaxis(tmp_clean_image_real, 1, -1)

                    output_clean_image_1[:, x:x + pat_size, y:y + pat_size, :] = output_clean_image_1[:, x:x + pat_size,
                                                                                 y:y + pat_size,
                                                                                 :] + tmp_clean_image_real



                    tmp_clean_image_imag = loaded_model.forward(imag_to_denoise,1).detach().numpy()
                    tmp_clean_image_imag=np.moveaxis(tmp_clean_image_imag, 1, -1)

                    output_clean_image_2[:, x:x + pat_size, y:y + pat_size, :] = output_clean_image_2[:, x:x + pat_size,
                                                                                 y:y + pat_size,
                                                                                 :] + tmp_clean_image_imag
                    count_image[:, x:x + pat_size, y:y + pat_size, :] = count_image[:, x:x + pat_size, y:y + pat_size,
                                                                        :] + np.ones((1, pat_size, pat_size, 1))


            output_clean_image_1 = output_clean_image_1 / count_image
            output_clean_image_2 = output_clean_image_2 / count_image
            output_clean_image = 0.5 * (np.square(denormalize_sar(output_clean_image_1)) + np.square(
            denormalize_sar(output_clean_image_2))) # combine the two estimation


            noisyimage = np.squeeze(np.sqrt(i_real_part ** 2 + i_imag_part ** 2))
            outputimage = np.sqrt(np.squeeze(output_clean_image))

            imagename = test_files[idx].replace(dataset_dir+"\\", "")
            print("Denoised image %s" % imagename)

            save_sar_images(outputimage, noisyimage, imagename, save_dir)
            save_real_imag_images(noisyimage, denormalize_sar(output_clean_image_1), denormalize_sar(output_clean_image_2),
                                  imagename, save_dir)
                                  
            save_real_imag_images_noisy(noisyimage, np.squeeze(i_real_part), np.squeeze(i_imag_part), imagename, save_dir)