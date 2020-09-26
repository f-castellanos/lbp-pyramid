import cv2
from .lbp import lbp
import pandas as pd
import numpy as np


class Preprocess:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.original_height = 584
        self.original_width = 565

    @staticmethod
    def read_img(path, flag=None):
        if flag is None:
            flag = cv2.IMREAD_COLOR
        return cv2.imread(path, flag)

    @staticmethod
    def apply_mask(img, mask):
        mask[mask < 15] = 0
        mask[mask > 15] = 1
        return cv2.bitwise_or(np.uint8(np.zeros(img.shape)), img, mask=mask)

    @staticmethod
    def resize(img, dim):
        """
        Resizes the image to the given dimensions.
        INTER_CUBIC – a bicubic interpolation over 4×4 pixel neighborhood
        :param img: original image
        :param dim: (width, height)
        :return: image resized
        """
        return cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)

    def crop(self, img, dim=(512, 512)):
        crop_height = (self.height - dim[0])/2
        if crop_height == int(crop_height):
            crop_bottom = int(crop_height)
            crop_top = self.height - int(crop_height)
        else:
            crop_bottom = int(crop_height)
            crop_top = self.height - (int(crop_height) + 1)
        crop_width = (self.width - dim[1])/2
        if crop_width == int(crop_width):
            crop_left = int(crop_width)
            crop_right = self.width - int(crop_width)
        else:
            crop_left = int(crop_width)
            crop_right = self.width - (int(crop_width) + 1)
        return img[crop_bottom:crop_top, crop_left:crop_right, :]

    def resize_add_borders(self, img):
        height, width = img.shape
        v_add = np.zeros((self.height - height, width))
        h_add = np.zeros((self.height, self.width - width))
        return np.float32(
            np.concatenate((
                np.concatenate((
                    v_add,
                    img
                ), axis=0),
                h_add
            ), axis=1)
        )

    @staticmethod
    def to_gray(img_bgr):
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def gray_to_frame(img):
        return pd.Series(img.reshape(1, -1)[0]).to_frame()

    def frame_to_img(self, df):
        return np.array(df).reshape(self.height, -1)

    # @staticmethod
    def apply_lbp(self, path, mask_path=None, plot=True):
        img = Preprocess.read_img(path, flag=0)
        if mask_path is not None:
            mask = Preprocess.read_img(path, flag=0)
            img = Preprocess.apply_mask(img, mask)
        img = self.resize_add_borders(img)
        img = lbp(img, plot=plot)
