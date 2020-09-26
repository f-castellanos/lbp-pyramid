import numpy as np
import matplotlib.pyplot as plt


class PlotResults:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def frame_to_img(self, df):
        return np.array(df).reshape(self.height, -1)

    def plot_label(self, df):
        label = df.loc[:, 'target']
        img = self.frame_to_img(label)
        plt.figure()
        plt.imshow(img, vmin=0, vmax=1, cmap='gray')
