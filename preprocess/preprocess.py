import cv2
from .lbp import lbp


class Preprocess:
    def __init__(self):
        pass

    @staticmethod
    def read_img(path):
        return cv2.imread(path)

    @staticmethod
    def apply_mask(img, mask):
        return cv2.bitwise_and(img, img, mask=mask)

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

    @staticmethod
    def to_gray(img_bgr):
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
