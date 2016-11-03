__author__ = 'MNR'

__all__ = ['RGB_to_BW', 'AC', 'extract_AC', 'import_AC']

import numpy as np
from scipy import misc
import scipy
import scipy.io


def RGB_to_BW(img):
    """
    converts RBG images to gray scale
    Parameters
    ----------
    img : 'Array'
        array of RGB pixel values

    Returns
    -------
        array of gray scale values
    """
    return np.dot(img, [0.299, 0.587, 0.114])

class AC(object):
    def __init__(self, xv, yv, cor):
        """
        Initiate new AC Class instance
        Parameters
        ----------
        xv : 'Array'
            array of x shifts
        yv : 'Array'
            array of y shifts
        cor : 'Array'
            array of auto correlation values

        Returns
        -------
        self.AC_2D : 'tuple'
            (xv, yv, cor)
        self.AD_1D : 'array'
            n x 2 array of (distance, auto correlation)
        self.h_sp : 'float'
            speckles size = distance at auto correlation = 0.5
        """

        self.AC_2D = (xv, yv, cor)

        dist = np.asarray([np.linalg.norm((x, y)) for x, y in zip(xv.reshape(-1), yv.reshape(-1))])
        img_AC = np.dstack((dist, cor.reshape(-1)))[0]

        img_AC = img_AC[img_AC[:, 0].argsort()]
        self.AC_1D = img_AC

        img_AC = img_AC[img_AC[:, 1].argsort()]
        h_sp = np.interp(0.5, img_AC[:, 1], img_AC[:, 0])*2
        self.h_sp = h_sp

class extract_AC(AC):
    def __init__(self, path, max_offset=None):
        """
        extracts xv, yv, cor from image file and initiates AC instance
        Parameters
        ----------
        path : 'string'
            image path
        max_offset : 'int' default = img_size/10
            maximum shift in pixels

        Returns
        -------
        self.path : 'string'
            image path
        """

        self.path = path
        img = misc.imread(path)
        img_size = img.shape

        if len(img_size) == 3:
            img = RGB_to_BW(img)
            img_size = img.shape

        if max_offset is None:
            max_offset = round(min(img_size)/10)

        img_size_offset = (img_size[0] - max_offset, img_size[1] - max_offset)
        offset_range = np.arange(-max_offset, max_offset+1)
        xv, yv = np.meshgrid(offset_range, offset_range)
        cor = np.ones((len(yv), len(xv)))

        plaquette = img[max_offset:img_size_offset[0], max_offset:img_size_offset[1]]

        for x in range(len(offset_range)):
            for y in range(len(offset_range)):
                x_offset = xv[y, x]
                y_offset = yv[y, x]

                offset_plaquette = img[max_offset + y_offset:img_size_offset[0] + y_offset, max_offset + x_offset:img_size_offset[1] + x_offset]
                cor[y, x] = np.corrcoef(plaquette.reshape(-1), offset_plaquette.reshape(-1))[0, 1]

        AC.__init__(self, xv, yv, cor)

    def export_to_mat(self, filename=None):
        """
        exports .mat file with auto correlation results
        Parameters
        ----------
        filename : 'string'
            name of .mat file, default is image file name

        Returns
        -------
        writes .mat file with (xv, yv, cor)
        """
        xv, yv, cor = self.AC_2D

        if filename is  None:
            filename = self.path.split('.')[0] + '.mat'

        scipy.io.savemat(filename, mdict={'x': xv, 'y': yv, 'AC': cor})

class import_AC(AC):
    def __init__(self, path):
        """
        extracts xv, yv, cor from .mat file and initiates AC instance
        Parameters
        ----------
        path : 'string'
            .mat file path

        Returns
        -------
        """
        AC_data = scipy.io.loadmat(path)
        xv = AC_data['x']
        yv = AC_data['y']
        cor = AC_data['AC']

        AC.__init__(self, xv, yv, cor)
