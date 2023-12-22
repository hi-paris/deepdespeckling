import struct
from osgeo import gdal
import numpy as np


def cos2mat(path_to_cosar_image: str) -> np.array:
    """Convert a CoSAR imge to a numpy array of size [ncolumns,nlines,2]

    Args:
        path_to_cosar_image (str): path to the image which is a cos file

    Returns:
        numpy array : the image in a numpy array
    """

    print('Converting CoSAR to numpy array of size [ncolumns,nlines,2]')

    try:
        fin = open(path_to_cosar_image, 'rb')
    except IOError:
        legx = path_to_cosar_image + ': it is a not openable file'
        print(legx)
        print(u'failed to call cos2mat')
        return 0, 0, 0, 0

    ibib = struct.unpack(">i", fin.read(4))[0]
    irsri = struct.unpack(">i", fin.read(4))[0]
    irs = struct.unpack(">i", fin.read(4))[0]
    ias = struct.unpack(">i", fin.read(4))[0]
    ibi = struct.unpack(">i", fin.read(4))[0]
    irtnb = struct.unpack(">i", fin.read(4))[0]
    itnl = struct.unpack(">i", fin.read(4))[0]

    nlig = struct.unpack(">i", fin.read(4))[0]
    ncoltot = int(irtnb / 4)
    ncol = ncoltot - 2
    nlig = ias

    print(u'Reading image in CoSAR format.  ncolumns=%d  nlines=%d' % (ncol, nlig))

    firm = np.zeros(4 * ncoltot, dtype=np.byte())
    imgcxs = np.empty([nlig, ncol], dtype=np.complex64())

    fin.seek(0)
    firm = fin.read(4 * ncoltot)
    firm = fin.read(4 * ncoltot)
    firm = fin.read(4 * ncoltot)
    firm = fin.read(4 * ncoltot)

    for iut in range(nlig):
        firm = fin.read(4 * ncoltot)
        imgligne = np.ndarray(2 * ncoltot, '>h', firm)
        imgcxs[iut, :] = imgligne[4:2 * ncoltot:2] + \
            1j * imgligne[5:2 * ncoltot:2]

    print('[:,:,0] contains the real part of the SLC image data')
    print('[:,:,1] contains the imaginary part of the SLC image data')

    return np.stack((np.real(imgcxs), np.imag(imgcxs)), axis=2)


def load_tiff_image(path_to_tiff_image: str) -> np.array:
    """Load a tiff image in a numpy array using gdal library

    Args:
        path_to_tiff_image (str): path to a tiff image

    Returns:
        (numpy array): numpy array containing the tiff image
    """
    dataset = gdal.Open(path_to_tiff_image)

    for x in range(1, dataset.RasterCount + 1):
        band = dataset.GetRasterBand(x)
        array = band.ReadAsArray()

    return array.astype(np.float64)
