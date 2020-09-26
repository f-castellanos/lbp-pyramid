import numpy as np
import matplotlib.pyplot as plt
import cv2


class PlotResults:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    @staticmethod
    def apply_red_mask(img, mask):
        img = np.stack([img, img, img], axis=2)
        mask = mask == 1
        img[mask] = [255, 0, 0]  # red patch in upper left
        return img

    def frame_to_img(self, df):
        return np.array(df).reshape(self.height, -1)

    def plot_label(self, df):
        img = self.frame_to_img(df.loc[:, 'target'])
        plt.figure()
        plt.imshow(img, vmin=0, vmax=1, cmap='gray')

    def plot_label_img(self, df):
        img = self.frame_to_img(df.loc[:, '1:1'])
        mask = self.frame_to_img(df.loc[:, 'target'])
        img = self.apply_red_mask(img, mask)
        plt.figure()
        plt.imshow(img, vmin=0, vmax=1, cmap='gray')
