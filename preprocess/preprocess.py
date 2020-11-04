from .lbp import lbp
import pandas as pd
import numpy as np
import PIL


class Preprocess:
    def __init__(self, height, width, lbp_radius=1, lbp_method='default', mask_threshold=15, label_threshold=30):
        self.height = height
        self.width = width
        self.original_height = 584
        self.original_width = 565
        self.mask_threshold = mask_threshold
        self.label_threshold = label_threshold
        self.lbp_radius = lbp_radius
        self.lbp_method = lbp_method

    @staticmethod
    def read_img(path):
        """
        Reads a image given the path
        :param path: Image path
        :return: Numpy array containing the information of the image
        """
        img = np.asarray(PIL.Image.open(path).convert('L'))
        return img.copy()

    def apply_mask(self, img, mask):
        """
        Applies the given mask to the given image. The pixels corresponding to small mask values are discarded.
        :param img: Numpy array containing the information of the image.
        :param mask: Numpy array containing the information of the mask.
        :return: Numpy array containing the information of the image after the mask has been applied.
        """
        # img[mask < self.mask_threshold] = np.median(img[mask > 15])
        img[mask < self.mask_threshold] = 0
        return img

    @staticmethod
    def rescale(img, dim):
        """
        Resizes the image to the given dimensions.
        PIL.Image.LANCZOS – Calculate the output pixel value using a high-quality Lanczos filter on
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

    def apply_lbp(self, img, plot=False):
        """
        Returns the LBP values for the given image
        :param img: Numpy array containing the information of the image.
        :param plot: Boolean to determine whether to plot the results
        :return: Numpy array containing the information of the LBP image.
        """
        img = lbp(img, r=self.lbp_radius, method=self.lbp_method, plot=plot)
        return img

    def remove_mask_data(self, dataset, mask):
        if mask is not None:
            mask_and_borders = self.rescale_add_borders(mask)
            return dataset[mask_and_borders.ravel() >= self.mask_threshold, :]
        else:
            borders = self.rescale_add_borders(
                np.ones((self.original_height, self.original_width)))
            return dataset[borders.ravel() == 1, :]

    def get_dataset(self, path, label_path=None, mask_path=None, balance=False, plot=False):
        """
        Returns the pyramid LBP values for each pixel of the given image
        :param balance: Boolean to determine whether the dataset must be balanced
        :param path: Image path
        :param label_path: Path of the label
        :param mask_path: Path of the mask
        :param plot: Boolean to determine whether to plot the results
        :return: DataFrame of the features calculated from the image
        """
        img = Preprocess.read_img(path)
        if mask_path is not None:
            mask = Preprocess.read_img(path)
            img = self.apply_mask(img, mask)
        else:
            mask = None
        img = self.rescale_add_borders(img)
        lbp_matrix = np.zeros((self.height * self.width, 6))
        for i in 2**np.arange(6):
            img_resized = Preprocess.rescale(img.copy(), (self.width//i, self.height//i))
            img_lbp = self.apply_lbp(img_resized, plot=plot)
            img_lbp = img_lbp[np.repeat(np.arange(img_lbp.shape[0]), i), :]
            img_lbp = img_lbp[:, np.repeat(np.arange(img_lbp.shape[1]), i)]
            lbp_matrix[:, int(np.log2(i))] = img_lbp.ravel()
        lbp_matrix = self.remove_mask_data(lbp_matrix, mask)
        if label_path is not None:
            label = self.get_label(label_path, mask)
            lbp_matrix = np.concatenate((lbp_matrix, label.reshape(-1, 1)), axis=1)
            if plot:
                self.plot_preprocess_with_label(lbp_matrix[:, 0], lbp_matrix[:, -1], mask)
            if balance:
                random_sample = np.random.choice(np.where(lbp_matrix[:, -1] == 0)[0],
                                                 size=sum(lbp_matrix[:, -1] == 1),
                                                 replace=False)
                sample = np.sort(
                    np.concatenate((random_sample, np.where(lbp_matrix[:, -1] == 1)[0]))
                )
                lbp_matrix = lbp_matrix[sample, :]
            df = pd.DataFrame(lbp_matrix, columns=['1:1', '1:2', '1:4', '1:8', '1:16', '1:32', 'label'], dtype='uint8')
        else:
            df = pd.DataFrame(lbp_matrix, columns=['1:1', '1:2', '1:4', '1:8', '1:16', '1:32'], dtype='uint8')
        return df

    def get_label(self, path, mask):
        img = Preprocess.read_img(path)
        img = Preprocess.gray_to_array(img).ravel()
        img[img < self.label_threshold] = 0
        img[img > self.label_threshold] = 1
        if mask is not None:
            img = img[mask.ravel() >= self.mask_threshold]
        return img

    def plot_preprocess_with_label(self, img, label, mask):
        def array_to_mat(arr, mask_mat):
            mat = np.copy(mask_mat)
            mat[mat < self.mask_threshold] = 0
            mat[mat >= self.mask_threshold] = arr
            return mat

        if mask is None:
            mask = np.zeros((self.original_height, self.original_width))
        img = array_to_mat(img, mask)
        label = array_to_mat(label, mask)
        img = np.asarray(PIL.Image.fromarray(np.uint8(img)).convert('RGB')).copy()
        img[label == 1] = [255, 0, 0]
        im = PIL.Image.fromarray(np.uint8(img))
        im.show()
