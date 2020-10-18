import cv2
from .lbp import lbp
import pandas as pd
import numpy as np
import PIL


class Preprocess:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.original_height = 584
        self.original_width = 565

    @staticmethod
    def read_img(path):
        """
        Reads a image given the path
        :param path: Image path
        :return: Numpy array containing the information of the image
        """
        img = np.asarray(PIL.Image.open(path).convert('L'))
        return img.copy()

    @staticmethod
    def apply_mask(img, mask):
        """
        Applies the given mask to the given image. The pixels corresponding to small mask values are discarded.
        :param img: Numpy array containing the information of the image.
        :param mask: Numpy array containing the information of the mask.
        :return: Numpy array containing the information of the image after the mask has been applied.
        """
        # img[mask < 15] = np.median(img[mask > 15])
        img[mask < 15] = 0
        return img

    @staticmethod
    def rescale(img, dim):
        """
        Resizes the image to the given dimensions.
        PIL.Image.LANCZOS – Calculate the output pixel value using a high-quality Lanczos filter (a truncated sinc) on
        all pixels that may contribute to the output value.
        https://pillow.readthedocs.io/en/stable/handbook/concepts.html
        :param img: original image
        :param dim: (width, height)
        :return: image resized
        """
        im = PIL.Image.fromarray(np.uint8(img))
        if img.shape != dim:
            return np.asarray(im.resize(dim, resample=PIL.Image.LANCZOS))
        else:
            return img

    def crop(self, img, dim=(512, 512)):
        crop_height = (self.height - dim[0]) / 2
        if crop_height == int(crop_height):
            crop_bottom = int(crop_height)
            crop_top = self.height - int(crop_height)
        else:
            crop_bottom = int(crop_height)
            crop_top = self.height - (int(crop_height) + 1)
        crop_width = (self.width - dim[1]) / 2
        if crop_width == int(crop_width):
            crop_left = int(crop_width)
            crop_right = self.width - int(crop_width)
        else:
            crop_left = int(crop_width)
            crop_right = self.width - (int(crop_width) + 1)
        return img[crop_bottom:crop_top, crop_left:crop_right, :]

    def rescale_add_borders(self, img):
        """
        Adds dummy pixels to the given image to rescale the image to the desired size.
        :param img: Numpy array containing the information of the image.
        :return: Numpy array containing the information of the resized image.
        """
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
    def gray_to_array(img):
        return img.ravel()

    @staticmethod
    def apply_lbp(img, plot=False):
        """
        Returns the LBP values for the given image
        :param img: Numpy array containing the information of the image.
        :param plot: Boolean to determine whether to plot the results
        :return: Numpy array containing the information of the LBP image.
        """
        img = lbp(img, plot=plot)
        return img

    def get_features(self, path, mask_path=None, plot=False):
        """
        Returns the pyramid LBP values for each pixel of the given image
        :param path: Image path
        :param mask_path: Path of the mask
        :param plot: Boolean to determine whether to plot the results
        :return: DataFrame of the features calculated from the image
        """
        img = Preprocess.read_img(path)
        if mask_path is not None:
            mask = Preprocess.read_img(path)
            img = Preprocess.apply_mask(img, mask)
        img = self.rescale_add_borders(img)
        lbp_matrix = np.zeros((self.height * self.width, 6))
        for i in 2**np.arange(6):
            img_resized = Preprocess.rescale(img.copy(), (self.width//i, self.height//i))
            img_lbp = Preprocess.apply_lbp(img_resized, plot=plot)
            img_lbp = img_lbp[np.repeat(np.arange(img_lbp.shape[0]), i), :]
            img_lbp = img_lbp[:, np.repeat(np.arange(img_lbp.shape[1]), i)]
            lbp_matrix[:, int(np.log2(i))] = img_lbp.ravel()
        df = pd.DataFrame(lbp_matrix, columns=['1:1', '1:2', '1:4', '1:8', '1:16', '1:32'], dtype='uint8')
        return df

    def get_label(self, path):
        img = Preprocess.read_img(path)
        img = self.rescale_add_borders(img)
        df = pd.DataFrame(Preprocess.gray_to_array(img), columns=['label']).astype(int)
        df[df < 30] = 0
        df[df > 30] = 1
        return df
