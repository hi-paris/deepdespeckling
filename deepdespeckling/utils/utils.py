import numpy as np
import cv2
import os
from PIL import Image
from scipy import signal
from pathlib import Path
from glob import glob

from deepdespeckling.utils.load_cosar import cos2mat, load_tiff_image
from deepdespeckling.utils.constants import M, m


def normalize_sar_image(image: np.array) -> np.array:
    """normalize a sar image store in  a numpy array

    Args:
        image (numpy array): the image to be normalized

    Returns:
        (numpy array): normalized image
    """
    return ((np.log(np.clip(image, 0, image.max())+1e-6)-m)/(M-m)).astype(np.float32)


def denormalize_sar_image(image: np.array) -> np.array:
    """Denormalize a sar image store in  a numpy array

    Args:
        image (numpy array): a sar image

    Raises:
        TypeError: raise an error if the image file is not a numpy array

    Returns:
        (numpy array): the image denormalized
    """
    if not isinstance(image, np.ndarray):
        raise TypeError('Please provide a numpy array')
    return np.exp((M - m) * (np.squeeze(image)).astype('float32') + m)


def denormalize_sar_image_sar2sar(image: np.array) -> np.array:
    """Denormalize a sar image store i  a numpy array

    Args:
        image (numpy array): a sar image

    Raises:
        TypeError: raise an error if the image file is not a numpy array

    Returns:
        (numpy array): the image denormalized
    """
    if not isinstance(image, np.ndarray):
        raise TypeError('Please provide a numpy array')
    return np.exp((np.clip(np.squeeze(image), 0, image.max()))*(M-m)+m)


def load_sar_image(image_path: str) -> np.array:
    """Load a SAR image in a numpy array, use cos2mat function if the file is a cos file, 
    load_tiff_image if the file is a tiff file

    Args:
        image_path (str) : absolute path to a SAR image (cos or npy file)

    Returns:
        image (numpy array) : the image of dimension [ncolumns,nlines,2]
    """
    if Path(image_path).suffix == ".npy":
        image = np.load(image_path)
    elif Path(image_path).suffix == ".cos":
        image = cos2mat(image_path)
    elif Path(image_path).suffix == ".tiff":
        image = load_tiff_image(image_path)
    else:
        raise ValueError("the image should be a cos, npy or tiff file")
    return image


def load_sar_images(file_list):
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
    if not isinstance(file_list, list):
        image = np.load(file_list)
        image = np.array(image).reshape(
            1, np.size(image, 0), np.size(image, 1), 2)
        return image
    data = []
    for file in file_list:
        image = np.load(file)
        data.append(np.array(image).reshape(
            1, np.size(image, 0), np.size(image, 1), 2))
    return data


def create_empty_folder_in_directory(destination_directory_path: str, folder_name: str = "processed_images") -> str:
    """Create an empty folder in a given directory

    Args:
        destination_directory_path (str): path pf the directory in which an empty folder is created if it doest not exist yet
        folder_name (str, optional): name of the folder to create. Defaults to "processed_images".

    Returns:
        processed_images_path: path of the created empty folder
    """
    processed_images_path = destination_directory_path + f'/{folder_name}'
    if not os.path.exists(processed_images_path):
        os.mkdir(processed_images_path)
    return processed_images_path


def preprocess_and_store_sar_images(sar_images_path: str, processed_images_path: str, model_name: str = "spotlight"):
    """Convert coSAR images to numpy arrays and store it in a specified path

    Args:
        sar_images_path (str): path of a folder containing coSAR images to be converted in numpy array
        processed_images_path (str): path of the folder where converted images are stored
        model_name (str): model name to be use for despeckling 
    """
    ext = "cos" if model_name in ["spotlight", "stripmap"] else "tiff"
    images_paths = glob(os.path.join(sar_images_path, "*.npy")) + \
        glob(os.path.join(sar_images_path, f"*.{ext}"))
    for image_path in images_paths:
        imagename = image_path.split('/')[-1].split('.')[0]
        if not os.path.exists(processed_images_path + '/' + imagename + '.npy'):
            image = load_sar_image(image_path)
            np.save(processed_images_path + '/' + imagename + '.npy', image)


def preprocess_and_store_sar_images_from_coordinates(sar_images_path: str, processed_images_path: str, coordinates_dict: dict, model_name: str = "spotlight"):
    """Convert specified areas of coSAR images to numpy arrays and store it in a specified path

    Args:
        sar_images_path (str): path of a folder containing coSAR images to be converted in numpy array
        processed_images_path (str): path of the folder where converted images are stored
        coordinates_dict (dict): dictionary containing pixel boundaries of the area to despeckle (x_start, x_end, y_start, y_end)
        model_name (str): model name to be use for despeckling. Default to "spotlight"
    """
    x_start = coordinates_dict["x_start"]
    x_end = coordinates_dict["x_end"]
    y_start = coordinates_dict["y_start"]
    y_end = coordinates_dict["y_end"]

    ext = "cos" if model_name in ["spotlight", "stripmap"] else "tiff"

    images_paths = glob(os.path.join(sar_images_path, "*.npy")) + \
        glob(os.path.join(sar_images_path, f"*.{ext}"))
    for image_path in images_paths:
        imagename = image_path.split('/')[-1].split('.')[0]
        if not os.path.exists(processed_images_path + '/' + imagename + '.npy'):
            image = load_sar_image(image_path)
            if ext == "cos":
                np.save(processed_images_path + '/' + imagename +
                        '.npy', image[x_start:x_end, y_start:y_end, :])
            else:
                np.save(processed_images_path + '/' + imagename +
                        '.npy', image[x_start:x_end, y_start:y_end])


def get_maximum_patch_size(kernel_size: int, patch_bound: int) -> int:
    """Get maximum manifold of a number lower than a bound

    Args:
        kernel_size (int): the kernel size of the trained model
        patch_bound (int): the maximum bound of the kernel size

    Returns:
        maximum_patch_size (int) : the maximum patch size
    """
    k = 1

    while kernel_size * k < patch_bound:
        k = k * 2

    maximum_patch_size = int(kernel_size * (k/2))

    return maximum_patch_size


def get_maximum_patch_size_from_image_dimensions(kernel_size: int, height: int, width: int) -> int:
    """Get the maximum patch size from the width and heigth and the kernel size of the model we use

    Args:
        kernel_size (int): the kernel size of the trained model
        height (int): the heigth of the image
        width (int): the width of the image

    Returns:
        maximum_patch_size (int) : the maximum patch size to use for despeckling
    """
    patch_bound = min(height, width)

    if patch_bound <= kernel_size:
        maximum_patch_size = kernel_size
    else:
        maximum_patch_size = get_maximum_patch_size(
            kernel_size=kernel_size, patch_bound=patch_bound)

    return maximum_patch_size


def symetrise_real_and_imaginary_parts(real_part: np.array, imag_part: np.array) -> tuple[np.array, np.array]:
    """Symetrise given real and imaginary parts to ensure MERLIN properties

    Args:
        real_part (numpy array): real part of the noisy image to symetrise
        imag_part (numpy array): imaginary part of the noisy image to symetrise 

    Returns:
        np.real(ima2), np.imag(ima2) (numpy array, numpy array): symetrised real and imaginary parts of a noisy image
    """
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

    return np.real(ima2), np.imag(ima2)


def preprocess_image(image: np.array, threshold: float) -> np.array:
    """Preprocess image by limiting pixel values with a threshold

    Args:
        image (numpy array): image to preprocess
        threshold (float): pixel value threshold 

    Returns:
        image (cv2 Image): image to be saved
    """
    image = np.clip(image, 0, threshold)
    image = image / threshold * 255
    image = Image.fromarray(image.astype('float64')).convert('L')

    return image


def save_image_to_png(image: np.array, threshold: int, image_full_path: str):
    """Save a given image to a png file in a given folder

    Args:
        image (numpy array): image to save 
        threshold (float): threshold of pixel values of the image to be saved in png
        image_full_path (str): full path of the image

    Raises:
        TypeError: if the image is not a numpy array
    """
    if not isinstance(image, np.ndarray):
        raise TypeError('Please provide a numpy array')

    image = preprocess_image(image, threshold=threshold)
    image.save(image_full_path.replace('npy', 'png'))


def save_image_to_npy_and_png(image: np.array, save_dir: str, prefix: str, image_name: str, threshold: float):
    """Save a given image to npy and png in a given folder

    Args:
        image (numpy array): image to save
        save_dir (str): path to the folder where to save the image
        prefix (str): prefix of the image file name
        image_name (str): name of the image file
        threshold (float): threshold of image pixel values used for png conversion
    """
    image_full_path = save_dir + prefix + image_name

    # Save image to npy file
    np.save(image_full_path, image)

    # Save image to png file
    save_image_to_png(image, threshold, image_full_path)


def compute_psnr(Shat: np.array, S: np.array) -> float:
    """Compute Peak Signal to Noise Ratio

    Args:
        Shat (numpy array): a SAR amplitude image
        S (numpy array): a reference SAR image

    Returns:
        res (float): psnr value
    """
    P = np.quantile(S, 0.99)
    res = 10 * np.log10((P ** 2) / np.mean(np.abs(Shat - S) ** 2))
    return res


def get_cropping_coordinates(image: np.array, destination_directory_path: str, model_name: str, fixed: bool = True):
    """Launch the crop tool to enable the user to select the subpart of the image to be despeckled

    Args:
        image (numpy aray): full image to be cropped 
        destination_directory_path (str): path of a folder to store the results
        model_name (str): model name to be use for despeckling. Default to "spotlight"
        fixed (bool, optional): whether the area of selection has a fixed size of not. Defaults to True.
    """
    image = preprocess_sar_image_for_cropping(image, model_name)
    full_image = image.copy()
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
            if fixed:
                if x_start > x_end and y_start > y_end:
                    tempxstart = x_start
                    tempystart = y_start

                    x_start = tempxstart - 32
                    x_end = tempxstart

                    y_start = tempystart - 32
                    y_end = tempystart

                elif x_start > x_end and y_start < y_end:
                    tempxstart = x_start
                    tempystart = y_start

                    x_start = tempxstart - 32
                    y_start = tempystart

                    x_end = tempxstart
                    y_end = tempystart + 32

                elif x_start < x_end and y_start > y_end:
                    tempxstart = x_start
                    tempystart = y_start

                    x_start = tempxstart
                    y_start = tempystart - 32
                    x_end = tempxstart + 32
                    y_end = tempystart

                else:
                    x_end = x_start + 32
                    y_end = y_start + 32
            else:
                if x_start > x_end and y_start > y_end:
                    tempx = x_start
                    x_start = x_end
                    x_end = tempx

                    tempy = y_start
                    y_start = y_end
                    y_end = tempy

                elif x_start > x_end and y_start < y_end:
                    tempxstart = x_start
                    tempystart = y_start
                    tempxend = x_end
                    tempyend = y_end

                    x_start = tempxend
                    y_start = tempystart
                    x_end = tempxstart
                    y_end = tempyend

                elif x_start < x_end and y_start > y_end:
                    tempxstart = x_start
                    tempystart = y_start
                    tempxend = x_end
                    tempyend = y_end

                    x_start = tempxstart
                    y_start = tempyend
                    x_end = tempxend
                    y_end = tempystart

            # cropping is finished
            cv2.rectangle(image, (x_start, y_start),
                          (x_end, y_end), (255, 0, 0), 2)
            cropping = False

            refPoint = [(x_start, y_start), (x_end, y_end)]

            if len(refPoint) == 2:  # when two points were found
                cropped_image = full_image[refPoint[0][1] * 8:refPoint[1][1]
                                           * 8, refPoint[0][0] * 8:refPoint[1][0] * 8]
                if fixed:
                    cropped_image = cv2.resize(cropped_image, (256, 256))
                else:
                    cropped_image = cv2.resize(
                        cropped_image, (8 * (x_end - x_start), 8 * (y_end - y_start)))
                cv2.imshow("Cropped", cropped_image)

                with open(destination_directory_path+'/cropping_coordinates.txt', 'w') as filehandle:
                    for listitem in refPoint:
                        filehandle.write(f'{listitem}\n')

    h, w, _ = image.shape
    # resizing image
    image = cv2.resize(image, (int(w / 8), int(h / 8)))
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mouse_crop)

    while True:
        i = image.copy()
        if not cropping:
            cv2.imshow("image", image)
        elif cropping:
            cv2.imshow("image", i)
            if not fixed:
                cv2.rectangle(i, (x_start, y_start),
                              (x_end, y_end), (255, 0, 0), 2)
        key = cv2.waitKey(10)
        if key == ord('q'):
            cv2.destroyAllWindows()
            return get_cropping_coordinates_from_file(destination_directory_path=destination_directory_path)


def get_cropping_coordinates_from_file(destination_directory_path: str) -> list:
    """Get cropping coordinates from a file where it's stored

    Args:
        destination_directory_path (str): path of the file in which the cropping coordinates are stored

    Returns:
        cropping_coordinates (list): list containing cropping coordinates
    """
    cropping_coordinates = []

    with open(destination_directory_path+'/cropping_coordinates.txt', 'r') as filehandle:
        for line in filehandle:
            # Remove linebreak which is the last character of the string
            curr_place = eval(line[:-1])
            # Add item to the list
            cropping_coordinates.append(curr_place)

    return cropping_coordinates


def crop_image(image: np.array, image_path: str, cropping_coordinates: list, model_name: str, processed_images_path: str):
    """Crop an image using given cropping coordinates and store the result in a given folder

    Args:
        image (numpy array): image to be cropped
        image_path (str): path of the image
        cropping_coordinates (list): list of coordinates of cropping, format [(x1, y1), (x2, y2)]
        model_name (str): name of the model (stripmap, spotlight or sar2sar)
        processed_images_path (str): path of the folder where to store the cropped image in npy format
    """
    if model_name in ["spotlight", "stripmap"]:
        image_real_part = image[:, :, 0]
        image_imaginary_part = image[:, :, 1]

        cropped_image_real_part = image_real_part[cropping_coordinates[0][1] * 8:cropping_coordinates[1][1] * 8,
                                                  cropping_coordinates[0][0] * 8:cropping_coordinates[1][0] * 8]
        cropped_image_imaginary_part = image_imaginary_part[cropping_coordinates[0][1] * 8:cropping_coordinates[1][1] * 8,
                                                            cropping_coordinates[0][0] * 8:cropping_coordinates[1][0] * 8]

        cropped_image_real_part = cropped_image_real_part.reshape(cropped_image_real_part.shape[0],
                                                                  cropped_image_real_part.shape[1], 1)
        cropped_image_imaginary_part = cropped_image_imaginary_part.reshape(cropped_image_imaginary_part.shape[0],
                                                                            cropped_image_imaginary_part.shape[1], 1)

        cropped_image = np.concatenate(
            (cropped_image_real_part, cropped_image_imaginary_part), axis=2)
    else:
        cropped_image = image[cropping_coordinates[0][1] * 8:cropping_coordinates[1][1] * 8,
                              cropping_coordinates[0][0] * 8:cropping_coordinates[1][0] * 8]

        cropped_image = cropped_image.reshape(
            cropped_image.shape[0], cropped_image.shape[1], 1)

    image_path_name = Path(image_path)
    np.save(processed_images_path + '/' + image_path_name.stem +
            '_cropped_to_denoise', cropped_image)


def preprocess_sar_image_for_cropping(image: np.array, model_name: str) -> np.array:
    """Preprocess image to use the cropping tool

    Args:
        image (numpy array): image from which we get cropping coordinates by using the cropping tool
        model_name (str): name of the model (stripmap, spotlight or sar2sar)

    Returns:
        image (cv2 image): image preprocessed for cropping
    """
    if model_name in ["spotlight", "stripmap"]:
        image_data_real = image[:, :, 0]
        image_data_imag = image[:, :, 1]
        image = np.squeeze(
            np.sqrt(np.square(image_data_real) + np.square(image_data_imag)))

    threshold = np.mean(image) + 3 * np.std(image)

    image = np.clip(image, 0, threshold)
    image = image / threshold * 255

    image = Image.fromarray(image.astype('float64')).convert('L')
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    return image
