
from scipy import ndimage
import numpy as np
import cv2
import DIC

__author__ = 'Michael Rossol'

__all__ = ["euclidean_dist", "RGB_to_BW", "img_processing", "img_load"]


def euclidean_dist(point1, point2):
    """
    distance between point1 and point2
    Parameters
    ----------
    point1 : 'tuple' or 'list'
        coordinates of point 1
    point2 : 'tuple' or 'list'
        coordinates of point 2

    Returns
    -------
    distance : 'float'
    """
    return np.sum((np.asarray(point1) - np.asarray(point2))**2)**(1/2)


def RGB_to_BW(img):
    """
    convert RGB image to gray scale
    Parameters
    ----------
    img : 'array'
        RGB image array n x m x 3

    Returns
    -------
        gray scale img array n x m
    """
    return np.dot(img, [0.299, 0.587, 0.114])


class img_processing(object):
    def __init__(self, img_path):
        """
        initiate class instance
        Parameters
        ----------
        img_path : 'string'
            file path for image

        Returns
        -------
        self.img : 'array'
            gray-scale image array
        self.size : 'tuple'
            image size
        """
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
        self.img = img
        self.size = img.shape

    def crop(self, ycrop=None, xcrop=None):
        """
        crop image
        Parameters
        ----------
        ycrop : 'tuple' or 'list'
            new y-image coordinates
        xcrop : 'tuple' or 'list'
            new x-image coordinates

        Returns
        -------
        self.img : 'array'
            new gray-scale image array
        self.size : 'tuple'
            new image size

        """
        img_crop = self.img

        if ycrop is not None:
            img_crop = img_crop[ycrop[0]:ycrop[1], :]

        if xcrop is not None:
            img_crop = img_crop[:, xcrop[0]:xcrop[1]]

        self.img = img_crop
        self.size = img_crop.shape

    def threshold(self, thresh=None, invert=True):
        """
        threshold images to binary
        Parameters
        ----------
        thresh : 'float', 'tuple', 'list'
            threshold method, None = 'Automatic', 'list'/'tuple' = adaptive,
            'float' = standard
        invert : 'boole'
            invert binary.

        Returns
        -------
        self.img_b : 'array'
            binary image array
        """
        if thresh is None:
            (t_ref, b_img) = cv2.threshold(self.img, 0, 1,
                                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            self.thresh = t_ref
        elif isinstance(thresh, (list, tuple)):
            b_img = cv2.adaptiveThreshold(self.img, 1,
                                          cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY, thresh[0],
                                          thresh[1])
        else:
            img_max = np.max(self.img)
            b_img = cv2.threshold(self.img, thresh, img_max,
                                  cv2.THRESH_BINARY)[1]
            b_img[b_img == img_max] = 1

        if invert:
            zeros = b_img == 0
            ones = b_img == 1
            b_img[zeros] = 1
            b_img[ones] = 0

        self.img_b = b_img

    def opening(self, size=(5, 5), binary_in=False):
        """
        morphological opening
        Parameters
        ----------
        size : 'tuple'
            kernel size
        binary_in : 'boole'
            run on binary

        Returns
        -------
        self.img_b : 'array'
            new binary array
        self.img : 'array'
            new img array

        """
        kernel = np.ones(size, np.uint8)

        if binary_in:
            opened = cv2.morphologyEx(self.img_b, cv2.MORPH_OPEN, kernel)
            self.img_b = opened
        else:
            opened = cv2.morphologyEx(self.img, cv2.MORPH_OPEN, kernel)
            self.img = opened

    def closing(self, size=(5, 5), binary_in=False):
        """
        morphological closing
        Parameters
        ----------
        size : 'tuple'
            kernel size
        binary_in : 'boole'
            run on binary

        Returns
        -------
        self.img_b : 'array'
            new binary array
        self.img : 'array'
            new img array

        """
        kernel = np.ones(size, np.uint8)

        if binary_in:
            closed = cv2.morphologyEx(self.img_b, cv2.MORPH_CLOSE, kernel)
            self.img_b = closed
        else:
            closed = cv2.morphologyEx(self.img, cv2.MORPH_CLOSE, kernel)
            self.img = closed

    def dilate(self, size=(5, 5), iterations=1, binary_in=False):
        """
        morphological dilate
        Parameters
        ----------
        size : 'tuple'
            kernel size
        iterations : 'int'
            number of times to run kernel
        binary_in : 'boole'
            run on binary

        Returns
        -------
        self.img_b : 'array'
            new binary array
        self.img : 'array'
            new img array

        """
        kernel = np.ones(size, np.uint8)

        if binary_in:
            dilation = cv2.dilate(self.img_b, kernel, iterations=iterations)
            self.img_b = dilation
        else:
            dilation = cv2.dilate(self.img, kernel, iterations=iterations)
            self.img = dilation

    def erode(self, size=(5, 5), iterations=1, binary_in=False):
        """
        morphological erode
        Parameters
        ----------
        size : 'tuple'
            kernel size
        iterations : 'int'
            number of times to run kernel
        binary_in : 'boole'
            run on binary

        Returns
        -------
        self.img_b : 'array'
            new binary array
        self.img : 'array'
            new img array

        """
        kernel = np.ones(size, np.uint8)

        if binary_in:
            erosion = cv2.erode(self.img_b, kernel, iterations=iterations)
            self.img_b = erosion
        else:
            erosion = cv2.erode(self.img, kernel, iterations=iterations)
            self.img = erosion

    def get_labels(self):
        """
        find clusters and extract center and size
        Parameters
        ----------

        Returns
        -------
        self.labels : 'list'
            list of cluster labels
        self.centers : 'list'
            list of cluster centers
        self.sizes : 'list'
            list of cluster sizes

        """
        b_img = self.img_b
        label_im, nb_labels = ndimage.label(b_img)

        center = np.asarray(ndimage.center_of_mass(b_img, label_im,
                            range(1, nb_labels + 1)))
        size = np.asarray(ndimage.sum(b_img, label_im,
                          range(1, nb_labels + 1)))

        self.labels = label_im
        self.centers = center
        self.sizes = size


def ext_len(img_p, centers):
    """
    extract virtual extensometer length
    Parameters
    ----------
    img_p : 'instance'
        image processing instance
    centers : 'tuple' or 'list'
        reference centers

    Returns
    -------
    L : 'float'
        virtual extensometer length
    [point1.tolist(), point2.tolist()]
        new center coordinates
    """
    img_p.get_labels()

    idx1 = DIC.nearest(img_p.centers, centers[0])
    point1 = img_p.centers[idx1]

    idx2 = DIC.nearest(img_p.centers, centers[1])
    point2 = img_p.centers[idx2]

    L = euclidean_dist(point1, point2)

    return L, [point1.tolist(), point2.tolist()]


def img_load(data, file):
    """
    exract image load
    Parameters
    ----------
    data : 'array'
        img, load data
    file : 'string'
        img file path
    Returns
    -------
    load : 'float'
        load for image file
    """
    numb = int((file[:-4].split('_'))[-1])
    load = data[np.where(data[:, 0] == numb)[0], 2]
    return load
