import numpy as np
from PIL import Image
import scipy.ndimage
from scipy import special
from scipy import signal


# DEFINE PARAMETERS OF SPECKLE AND NORMALIZATION FACTOR
M = 10.089038980848645
m = -1.429329123112601
L = 1
c = (1 / 2) * (special.psi(L) - np.log(L))
cn = c / (M - m)  # normalized (0,1) mean of log speckle

def normalize_sar(im):
    """ Description
            ----------
            Normalization of a numpy-stored image

            Parameters
            ----------
            im : an image object

            Returns
            ----------

        """
    return ((np.log(im + np.spacing(1)) - m) * 255 / (M - m)).astype('float32')

def denormalize_sar(im):
    """ Description
            ----------
            Denormalization of a numpy image

            Parameters
            ----------
            im : an image object

            Returns
            ----------

        """
    return np.exp((M - m) * (np.squeeze(im)).astype('float32') + m)

def symetrisation_patch_test(real_part,imag_part):
    S = np.fft.fftshift(np.fft.fft2(real_part[0,:,:,0]+1j*imag_part[0,:,:,0]))
    p = np.zeros((S.shape[0])) # azimut (ncol)
    for i in range(S.shape[0]):
        p[i] = np.mean(np.abs(S[i,:]))
    sp = p[::-1]
    c = np.real(np.fft.ifft(np.fft.fft(p)*np.conjugate(np.fft.fft(sp))))
    d1 = np.unravel_index(c.argmax(),p.shape[0])
    d1 = d1[0]
    shift_az_1 = int(round(-(d1-1)/2))%p.shape[0]+int(p.shape[0]/2)
    p2_1 = np.roll(p,shift_az_1)
    shift_az_2 = int(round(-(d1-1-p.shape[0])/2))%p.shape[0]+int(p.shape[0]/2)
    p2_2 = np.roll(p,shift_az_2)
    window = signal.gaussian(p.shape[0], std=0.2*p.shape[0])
    test_1 = np.sum(window*p2_1)
    test_2 = np.sum(window*p2_2)
    # make sure the spectrum is symetrized and zeo-Doppler centered
    if test_1>=test_2:
        p2 = p2_1
        shift_az = shift_az_1/p.shape[0]
    else:
        p2 = p2_2
        shift_az = shift_az_2/p.shape[0]
    S2 = np.roll(S,int(shift_az*p.shape[0]),axis=0)

    q = np.zeros((S.shape[1])) # range (nlin)
    for j in range(S.shape[1]):
        q[j] = np.mean(np.abs(S[:,j]))
    sq = q[::-1]
    #correlation
    cq = np.real(np.fft.ifft(np.fft.fft(q)*np.conjugate(np.fft.fft(sq))))
    d2 = np.unravel_index(cq.argmax(),q.shape[0])
    d2=d2[0]
    shift_range_1 = int(round(-(d2-1)/2))%q.shape[0]+int(q.shape[0]/2)
    q2_1 = np.roll(q,shift_range_1)
    shift_range_2 = int(round(-(d2-1-q.shape[0])/2))%q.shape[0]+int(q.shape[0]/2)
    q2_2 = np.roll(q,shift_range_2)
    window_r = signal.gaussian(q.shape[0], std=0.2*q.shape[0])
    test_1 = np.sum(window_r*q2_1)
    test_2 = np.sum(window_r*q2_2)
    if test_1>=test_2:
        q2 = q2_1
        shift_range = shift_range_1/q.shape[0]
    else:
        q2 = q2_2
        shift_range = shift_range_2/q.shape[0]


    Sf = np.roll(S2,int(shift_range*q.shape[0]),axis=1)
    ima2 = np.fft.ifft2(np.fft.ifftshift(Sf))
    ima2 = ima2.reshape(1, np.size(ima2, 0), np.size(ima2, 1), 1)
    return np.real(ima2), np.imag(ima2)

def load_sar_images(filelist):
    """ Description
            ----------
            Loads files , resize them and append them into a list called data

            Parameters
            ----------
            filelist : a path to a folder containing the images

            Returns
            ----------
            A list of images

        """
    if not isinstance(filelist, list):
        im = np.load(filelist)
        return np.array(im).reshape(1, np.size(im, 0), np.size(im, 1), 2)
    data = []
    for file in filelist:
        im = np.load(file)
        data.append(np.array(im).reshape(1, np.size(im, 0), np.size(im, 1), 2))
    return data

def store_data_and_plot(im, threshold, filename):
    """ Description
            ----------
            Creates an image memory from an object exporting the array interface and returns a
            converted copy of this image into greyscale mode ("L")

            However, there is not plotting functions' call ?

            Parameters
            ----------
            im : the image to store
            threshold: clip a maximum value in the image array i.e values are to be between 0 and threshold
            filename: the path to store the result array image in .png

            Returns
            ----------
            None

        """
    im = np.clip(im, 0, threshold)
    im = im / threshold * 255
    im = Image.fromarray(im.astype('float64')).convert('L')
    im.save(filename.replace('npy','png'))


def save_sar_images(denoised, noisy, imagename, save_dir):
    """ Description
            ----------
            File management tool : saves both noisy and denoised images in corresponding folders

            Parameters
            ----------
            denoised : Array of denoised data to be saved
            noisy : Array of noisy  data to be saved
            imagename : Name of image
            save_dir : repository here to store both denoised and noisy images' folders

            Returns
            ----------
            None

        """
    choices = {'Serreponcon': 450.0, 'Sendai':600.0, 'Paris': 1291.0, 'Berlin': 1036.0, 'Bergen': 553.71,
          'SDP_Lambesc': 349.53, 'Grand_Canyon': 287.0}
    threshold = None
    for x in choices:
        if x in imagename:
            threshold = choices.get(x)
    if threshold is None: threshold = np.mean(noisy) + 3 * np.std(noisy)

    denoisedfilename = save_dir + "/denoised_" + imagename
    np.save(denoisedfilename, denoised)
    store_data_and_plot(denoised, threshold, denoisedfilename)

    noisyfilename = save_dir + "/noisy_" + imagename
    np.save(noisyfilename, noisy)
    store_data_and_plot(noisy, threshold, noisyfilename)

def save_real_imag_images(noisy, real_part, imag_part, imagename, save_dir):
    """ Description
            ----------
            Saving both real and imaginary parts of a given image

            Parameters
            ----------
            noisy :
            real_part :
            imag_part :
            imagename :
            save_dir :

            Returns
            ----------
            None

        """
    choices = {'Serreponcon': 450.0, 'Sendai':600.0, 'Paris': 1291.0, 'Berlin': 1036.0, 'Bergen': 553.71,
          'SDP_Lambesc': 349.53, 'Grand_Canyon': 287.0}
    threshold = None
    for x in choices:
        if x in imagename:
            threshold = choices.get(x)
    if threshold is None: threshold = np.mean(noisy) + 3 * np.std(noisy)

    realfilename = save_dir + "/denoised_real_" + imagename
    np.save(realfilename, real_part)
    store_data_and_plot(real_part, threshold, realfilename)

    imagfilename = save_dir + "/denoised_imag_" + imagename
    np.save(imagfilename, imag_part)
    store_data_and_plot(imag_part, threshold, imagfilename)

def save_real_imag_images_noisy(noisy, real_part, imag_part, imagename, save_dir):
    choices = {'Serreponcon': 450.0, 'Sendai':600.0, 'Paris': 1291.0, 'Berlin': 1036.0, 'Bergen': 553.71,
          'SDP_Lambesc': 349.53, 'Grand_Canyon': 287.0}
    threshold = None
    for x in choices:
        if x in imagename:
            threshold = choices.get(x)
    if threshold is None: threshold = np.mean(noisy) + 3 * np.std(noisy)

    realfilename = save_dir + "/noisy_real_" + imagename
    np.save(realfilename, real_part)
    store_data_and_plot(np.sqrt(2)*np.abs(real_part), threshold, realfilename)

    imagfilename = save_dir + "/noisy_imag_" + imagename
    np.save(imagfilename, imag_part)
    store_data_and_plot(np.sqrt(2)*np.abs(imag_part), threshold, imagfilename)
