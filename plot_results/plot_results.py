import numpy as np
import matplotlib.pyplot as plt
import PIL


class PlotResults:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    @staticmethod
    def apply_red_mask(img, mask):
        img = np.asarray(PIL.Image.fromarray(np.uint8(img)).convert('RGB')).copy()
        img[mask == 1] = [255, 0, 0]  # red patch in upper left
        return img

    def frame_to_img(self, df):
        return np.array(df).reshape(self.height, self.width)

    def plot_label(self, df):
        img = self.frame_to_img(df.loc[:, 'label'])
        plt.figure()
        plt.imshow(img, vmin=0, vmax=1, cmap='gray')

    def plot_label_img(self, df):
        img = self.frame_to_img(df.loc[:, '1:1'])
        mask = self.frame_to_img(df.loc[:, 'label'])
        img = self.apply_red_mask(img, mask)
        im = PIL.Image.fromarray(np.uint8(img))
        im.show()
