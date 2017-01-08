"""
Scripts to extract frontal velocity from image sequences
"""
import os
import numpy as np
import copy
import cv2

__author__ = 'MNR'

__all__ = ['fit_line', 'img_process', 'delete_duplicates']


def fit_line(data):
    assert data.shape[1] == 2, "Data is not an nx2 array."
    x = data[:, 0]
    y = data[:, 1]
    A = np.vstack([x, np.ones(len(y))]).T
    return np.linalg.lstsq(A, y)[0]


class img_process(object):
    def __init__(self, img_path):
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
        self.img = img

    def threshold(self, thresh=None):
        if thresh is None:
            (t_ref, b_img) = cv2.threshold(self.img, 0, 1,
                                           cv2.THRESH_BINARY_INV |
                                           cv2.THRESH_OTSU)
            self.thresh = t_ref
        elif isinstance(thresh, tuple):
            b_img = cv2.adaptiveThreshold(self.img, 1,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, thresh[0],
                                          thresh[1])
        else:
            img_max = np.max(self.img)
            b_img = cv2.threshold(self.img, thresh, img_max,
                                  cv2.THRESH_BINARY_INV)[1]
            b_img[b_img == img_max] = 1

        self.img_b = b_img

    def flip(self):
        img = copy.copy(self.img)
        img_b = copy.copy(self.img_b)

        self.img = img[:, ::-1]
        self.img_b = img_b[:, ::-1]

    def crop(self, ycrop=None, xcrop=None):

        img_crop = self.img
        b_crop = self.img_b

        if ycrop is not None:
            img_crop = img_crop[ycrop[0]:ycrop[1], :]
            b_crop = b_crop[ycrop[0]:ycrop[1], :]

        if xcrop is not None:
            img_crop = img_crop[:, xcrop[0]:xcrop[1]]
            b_crop = b_crop[:, xcrop[0]:xcrop[1]]

        self.img_c = img_crop
        self.img_b = b_crop

    def erode(self, erode_size=(15, 1), erode_n=1, binary_in=True):
        kernel = np.ones(erode_size, np.uint8)

        if binary_in:
            erosion = cv2.erode(self.img_b, kernel, iterations=erode_n)
            self.img_b = erosion
        else:
            erosion = cv2.erode(self.img, kernel, iterations=erode_n)
            self.img = erosion

    def find_front(self, direction='left'):

        ypos, xpos = np.where(self.img_b == 1)
        yedge, xedge = np.shape(self.img_b)

        if ypos.shape[0] > 0:
            min_pos = xpos.argmin()
            xmin, ymin = xpos[min_pos], ypos[min_pos]
            if xmin == 0:
                xmin = np.nan
            if ymin == 0:
                ymin = np.nan

            max_pos = xpos.argmax()
            xmax, ymax = xpos[max_pos], ypos[max_pos]
            if xmax == xedge - 1:
                xmax = np.nan
            if ymax == yedge - 1:
                ymax = np.nan
        else:
            xmin, xmax, ymin, ymax = np.nan, np.nan, np.nan, np.nan

        if direction.lower().startswith('l'):
            self.ypos = ymin
            self.xpos = xmin
        else:
            self.ypos = ymax
            self.xpos = xmax


def delete_duplicates(path):
    for file in os.listdir(path):
        if file.endswith('(1).png'):
            os.remove(os.path.join(path, file))
