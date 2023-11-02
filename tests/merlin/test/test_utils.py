import numpy as np
from PIL import Image
from scipy import special
from scipy import signal
from deepdespeckling.merlin.inference.load_cosar import cos2mat
import cv2
from PIL import Image
import numpy as np
from numpy import asarray


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
    if not isinstance(im, np.ndarray):
        raise TypeError('Please provide a .npy argument')
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
    if not isinstance(im, np.ndarray):
        raise TypeError('Please provide a .npy argument')
    return np.exp((M - m) * (np.squeeze(im)).astype('float32') + m)


def symetrisation_patch_test(real_part, imag_part):
    if not isinstance(real_part, np.ndarray):
        raise TypeError('Please provide a .npy argument')
    if not isinstance(imag_part, np.ndarray):
        raise TypeError('Please provide a .npy argument')
    S = np.fft.fftshift(np.fft.fft2(
        real_part[0, :, :, 0] + 1j * imag_part[0, :, :, 0]))
    p = np.zeros((S.shape[0]))  # azimut (ncol)
    for i in range(S.shape[0]):
        p[i] = np.mean(np.abs(S[i, :]))
    sp = p[::-1]
    c = np.real(np.fft.ifft(np.fft.fft(p) * np.conjugate(np.fft.fft(sp))))
    d1 = np.unravel_index(c.argmax(), p.shape[0])
    d1 = d1[0]
    shift_az_1 = int(round(-(d1 - 1) / 2)) % p.shape[0] + int(p.shape[0] / 2)
    p2_1 = np.roll(p, shift_az_1)
    shift_az_2 = int(
        round(-(d1 - 1 - p.shape[0]) / 2)) % p.shape[0] + int(p.shape[0] / 2)
    p2_2 = np.roll(p, shift_az_2)
    window = signal.gaussian(p.shape[0], std=0.2 * p.shape[0])
    test_1 = np.sum(window * p2_1)
    test_2 = np.sum(window * p2_2)
    # make sure the spectrum is symetrized and zeo-Doppler centered
    if test_1 >= test_2:
        p2 = p2_1
        shift_az = shift_az_1 / p.shape[0]
    else:
        p2 = p2_2
        shift_az = shift_az_2 / p.shape[0]
    S2 = np.roll(S, int(shift_az * p.shape[0]), axis=0)

    q = np.zeros((S.shape[1]))  # range (nlin)
    for j in range(S.shape[1]):
        q[j] = np.mean(np.abs(S[:, j]))
    sq = q[::-1]
    # correlation
    cq = np.real(np.fft.ifft(np.fft.fft(q) * np.conjugate(np.fft.fft(sq))))
    d2 = np.unravel_index(cq.argmax(), q.shape[0])
    d2 = d2[0]
    shift_range_1 = int(round(-(d2 - 1) / 2)
                        ) % q.shape[0] + int(q.shape[0] / 2)
    q2_1 = np.roll(q, shift_range_1)
    shift_range_2 = int(
        round(-(d2 - 1 - q.shape[0]) / 2)) % q.shape[0] + int(q.shape[0] / 2)
    q2_2 = np.roll(q, shift_range_2)
    window_r = signal.gaussian(q.shape[0], std=0.2 * q.shape[0])
    test_1 = np.sum(window_r * q2_1)
    test_2 = np.sum(window_r * q2_2)
    if test_1 >= test_2:
        q2 = q2_1
        shift_range = shift_range_1 / q.shape[0]
    else:
        q2 = q2_2
        shift_range = shift_range_2 / q.shape[0]

    Sf = np.roll(S2, int(shift_range * q.shape[0]), axis=1)
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
    if not isinstance(im, np.ndarray):
        raise TypeError('Please provide a .npy argument')
    im = np.clip(im, 0, threshold)

    im = im / threshold * 255
    im = Image.fromarray(im.astype('float64')).convert('L')
    im.save(filename.replace('npy', 'png'))
    return filename


def save_sar_images(denoised, noisy, imagename, save_dir, groundtruth=None):
    choices = {'marais1': 190.92, 'marais2': 168.49, 'saclay': 470.92, 'lely': 235.90, 'ramb': 167.22,
               'risoul': 306.94, 'limagne': 178.43, 'saintgervais': 560, 'Serreponcon': 450.0,
               'Sendai': 600.0, 'Paris': 1291.0, 'Berlin': 1036.0, 'Bergen': 553.71,
               'SDP_Lambesc': 349.53, 'Grand_Canyon': 287.0, 'domancy': 560, 'Brazil': 103.0}
    threshold = None
    for x in choices:
        if x in imagename:
            threshold = choices.get(x)
    if threshold is None:
        threshold = np.mean(noisy) + 3 * np.std(noisy)

    ####
    imagename = imagename.split('\\')[-1]
    ####

    if groundtruth:
        groundtruthfilename = save_dir + "/groundtruth_" + imagename
        np.save(groundtruthfilename, groundtruth)
        store_data_and_plot(groundtruth, threshold, groundtruthfilename)

    denoisedfilename = save_dir + "/denoised_" + imagename
    np.save(denoisedfilename, denoised)
    store_data_and_plot(denoised, threshold, denoisedfilename)

    noisyfilename = save_dir + "/noisy_" + imagename
    np.save(noisyfilename, noisy)
    store_data_and_plot(noisy, threshold, noisyfilename)


def save_real_imag_images(real_part, imag_part, imagename, save_dir):
    choices = {'marais1': 190.92, 'marais2': 168.49, 'saclay': 470.92, 'lely': 235.90, 'ramb': 167.22,
               'risoul': 306.94, 'limagne': 178.43, 'saintgervais': 560, 'Serreponcon': 450.0,
               'Sendai': 600.0, 'Paris': 1291.0, 'Berlin': 1036.0, 'Bergen': 553.71,
               'SDP_Lambesc': 349.53, 'Grand_Canyon': 287.0, 'Brazil': 103.0}
    threshold = None
    for x in choices:
        if x in imagename:
            threshold = choices.get(x)
    if threshold is None:
        threshold = np.mean(imag_part) + 3 * np.std(imag_part)

    ####
    imagename = imagename.split('\\')[-1]
    ####

    realfilename = save_dir + "/denoised_real_" + imagename
    np.save(realfilename, real_part)
    store_data_and_plot(real_part, threshold, realfilename)

    imagfilename = save_dir + "/denoised_imag_" + imagename
    np.save(imagfilename, imag_part)
    store_data_and_plot(imag_part, threshold, imagfilename)


def save_real_imag_images_noisy(real_part, imag_part, imagename, save_dir):
    choices = {'marais1': 190.92, 'marais2': 168.49, 'saclay': 470.92, 'lely': 235.90, 'ramb': 167.22,
               'risoul': 306.94, 'limagne': 178.43, 'saintgervais': 560, 'Serreponcon': 450.0,
               'Sendai': 600.0, 'Paris': 1291.0, 'Berlin': 1036.0, 'Bergen': 553.71,
               'SDP_Lambesc': 349.53, 'Grand_Canyon': 287.0, 'Brazil': 103.0}
    threshold = None
    for x in choices:
        if x in imagename:
            threshold = choices.get(x)
    if threshold is None:
        threshold = np.mean(np.abs(imag_part)) + 3 * np.std(np.abs(imag_part))

    ####
    imagename = imagename.split('\\')[-1]
    ####

    realfilename = save_dir + "/noisy_real_" + imagename
    np.save(realfilename, real_part)
    store_data_and_plot(np.sqrt(2) * np.abs(real_part),
                        threshold, realfilename)

    imagfilename = save_dir + "/noisy_imag_" + imagename
    np.save(imagfilename, imag_part)
    store_data_and_plot(np.sqrt(2) * np.abs(imag_part),
                        threshold, imagfilename)


def cal_psnr(Shat, S):
    # takes amplitudes in input
    # Shat: a SAR amplitude image
    # S:    a reference SAR image
    P = np.quantile(S, 0.99)
    res = 10 * np.log10((P ** 2) / np.mean(np.abs(Shat - S) ** 2))
    return res


def crop(image_png, image_data_real, image_data_imag, destination_directory, test_data):
    """ A crapping tool for despeckling only the selection of the user, made with OpenCV
            Parameters
            ----------
            image_png: .png file
            the image to be cropped in png format
            image_data_real: nd.array
            the real part of the image stored in an array
            image_data_imag: nd.array
            the imaginary part of the image stored in an array
            destination_directory: string
            the path for saving results in
            test_data: string
            the path for saving results in
            cropping: bool
            A boolean stating if the user wants to crop the image or not
            Returns
            ----------
            None
        """
    # HERE I READ THE PNG FILE
    oriImage = image_png.copy()
    cropping = False
    x_start, y_start, x_end, y_end = 0, 0, 0, 0

    # CV2 CROPPING IN WINDOW
    def mouse_crop(event, x, y, flags, param):
        """ The callback function of crop() to deal with user's events
        """
        global x_start, y_start, x_end, y_end, cropping
        cropping = False

        if event == cv2.EVENT_LBUTTONDOWN:
            x_start, y_start, x_end, y_end = x, y, x, y
            cropping = True

        # Mouse is Moving
        elif event == cv2.EVENT_MOUSEMOVE:
            x_end, y_end = x, y

        # if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates
            x_end, y_end = x, y
            # cropping is finished
            cv2.rectangle(image, (x_start, y_start), (x, y), (255, 0, 0), 2)
            cropping = False

            refPoint = [(x_start, y_start), (x_end, y_end)]

            if len(refPoint) == 2:  # when two points were found
                image_data_real_cropped = image_data_real[refPoint[0][1] * 8:refPoint[1][1] * 8,
                                                          refPoint[0][0] * 8:refPoint[1][0] * 8]
                image_data_imag_cropped = image_data_imag[refPoint[0][1] * 8:refPoint[1][1] * 8,
                                                          refPoint[0][0] * 8:refPoint[1][0] * 8]

                roi = oriImage[refPoint[0][1] * 8:refPoint[1][1]
                               * 8, refPoint[0][0] * 8:refPoint[1][0] * 8]
                roi = cv2.resize(
                    roi, (8 * (x_end - x_start), 8 * (y_end - y_start)))

                cv2.imwrite(destination_directory +
                            '\\cropped_npy_to_png.png', roi)
                cv2.imshow("Cropped", roi)
                cropped_img_png = Image.open(
                    destination_directory + '\\cropped_npy_to_png.png')
                numpy_crop = asarray(cropped_img_png)
                np.save(destination_directory + '\\cropped.npy', numpy_crop)
                np.save(test_data + '\\image_data_real_cropped.npy',
                        image_data_real_cropped)
                np.save(test_data + '\\image_data_imag_cropped.npy',
                        image_data_imag_cropped)

    h, w, c = image_png.shape
    # resizing image
    image = cv2.resize(image_png, (int(w / 8), int(h / 8)))
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mouse_crop)

    while True:
        i = image.copy()

        if not cropping:
            cv2.imshow("image", image)

        elif cropping:
            cv2.imshow("image", i)
            cv2.rectangle(i, (x_start, y_start),
                          (x_end, y_end), (255, 0, 0), 2)

        key = cv2.waitKey(10)
        if key == ord('q'):
            cv2.destroyAllWindows()
            return


def crop_fixed(image_png, image_data_real, image_data_imag, destination_directory, test_data):
    """ A crapping tool for despeckling only the selection of the user, made with OpenCV
            Parameters
            ----------
            image_png: .png file
            the image to be cropped in png format
            image_data_real: nd.array
            the real part of the image stored in an array
            image_data_imag: nd.array
            the imaginary part of the image stored in an array
            destination_directory: string
            the path for saving results in
            test_data: string
            the path for saving results in
            cropping: bool
            A boolean stating if the user wants to crop the image or not
            Returns
            ----------
            None
        """
    # HERE I READ THE PNG FILE
    oriImage = image_png.copy()
    cropping = False
    x_start, y_start, x_end, y_end = 0, 0, 0, 0

    # CV2 CROPPING IN WINDOW
    def mouse_crop_fixed(event, x, y, flags, param):
        """ The callback function of crop() to deal with user's events
        """
        global x_start, y_start, x_end, y_end, cropping
        cropping = False
        if event == cv2.EVENT_LBUTTONDOWN:
            x_start, y_start, x_end, y_end = x, y, x, y
            cropping = True

        # Mouse is Moving
        elif event == cv2.EVENT_MOUSEMOVE:
            cv2.rectangle(image, (x_start, y_start),
                          (x_start + 32, y_start + 32), (255, 0, 0), 2)
            x_end, y_end = x, y

        # if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates
            x_end, y_end = x_start + 32, y_start + 32
            cropping = False  # cropping is finished

            refPoint = [(x_start, y_start), (x_end, y_end)]

            if len(refPoint) == 2:  # when two points were found
                image_data_real_cropped = image_data_real[refPoint[0][1] * 8:refPoint[1][1] * 8,
                                                          refPoint[0][0] * 8:refPoint[1][0] * 8]
                image_data_imag_cropped = image_data_imag[refPoint[0][1] * 8:refPoint[1][1] * 8,
                                                          refPoint[0][0] * 8:refPoint[1][0] * 8]
                roi = oriImage[refPoint[0][1] * 8:refPoint[1][1]
                               * 8, refPoint[0][0] * 8:refPoint[1][0] * 8]
                roi = cv2.resize(roi, (256, 256))
                cv2.imwrite(destination_directory +
                            '\\cropped_npy_to_png.png', roi)
                cv2.imshow("Cropped", roi)
                cropped_img_png = Image.open(
                    destination_directory + '\\cropped_npy_to_png.png')
                numpy_crop = asarray(cropped_img_png)
                np.save(destination_directory + '\\cropped.npy', numpy_crop)
                np.save(test_data + '\\image_data_real_cropped.npy',
                        image_data_real_cropped)
                np.save(test_data + '\\image_data_imag_cropped.npy',
                        image_data_imag_cropped)

    h, w, c = image_png.shape
    # resizing image
    image = cv2.resize(image_png, (int(w / 8), int(h / 8)))
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mouse_crop_fixed)

    while True:
        i = image.copy()

        if not cropping:
            cv2.imshow("image", image)

        elif cropping:
            cv2.imshow("image", i)

        key = cv2.waitKey(10)
        if key == ord('q'):
            cv2.destroyAllWindows()
            return


def get_info_image(image_path, destination_directory):
    """ A function for retrieving informations on the CoSar stored image such as its png equivalent, its real and
        imaginary part and the threshold to be applied later
            Parameters
            ----------
            image_path: string
            the path leading to image in CoSar format
            destination_directory: string
            the path for saving results in
            Returns
            ----------
            None
    """
    image_data = cos2mat(image_path)

    # GET THE TWO PARTS
    image_data_real = image_data[:, :, 0]
    image_data_imag = image_data[:, :, 1]

    # GET NOISY FOR THRESHOLD
    image = np.squeeze(
        np.sqrt(np.square(image_data_real) + np.square(image_data_imag)))
    threshold = np.mean(image) + 3 * np.std(image)

    # DISPLAY FULL PICTURE
    filename = store_data_and_plot(
        image, threshold, destination_directory + '\\image_provided.npy')
    print('full picture in png is saved')
    image_png = cv2.imread(filename.replace('npy', 'png'))
    print('full picture in png has a dimension of {size}'.format(
        size=image_png.shape))
    return image_png, image_data, image_data_real, image_data_imag, threshold, filename
