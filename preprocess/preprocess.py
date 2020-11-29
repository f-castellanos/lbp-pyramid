from .lbp import lbp
import pandas as pd
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import matplotlib.pyplot as plt
import cv2


class Preprocess:
    def __init__(self, height, width, lbp_radius=1, lbp_method='default', mask_threshold=100, label_threshold=30):
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
        img = np.asarray(Image.open(path).convert('L'))
        return img.copy()

    def filter_by_mask(self, img, mask_path):
        """
        Cancels the points that correspond to small values of the mask.
        :param img: Numpy array containing the information of the image.
        :param mask_path: str of the path of the mask.
        :return: Numpy array containing the information of the image after the mask has been applied.
        """
        if mask_path is not None:
            mask = Preprocess.read_img(mask_path)
            # img[mask < self.mask_threshold] = np.median(img[mask >= self.mask_threshold])
            img[mask < self.mask_threshold] = 0
        else:
            mask = None
        return img, mask

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
        im = Image.fromarray(np.uint8(img))
        if img.shape != dim:
            return np.asarray(im.resize(dim, resample=Image.LANCZOS))
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

    def remove_mask_data(self, arr, mask, remove_borders=False):
        if mask is not None:
            if remove_borders:
                mask_and_borders = self.rescale_add_borders(mask)
                return (arr[mask_and_borders.ravel() >= self.mask_threshold, :],
                        np.where(mask_and_borders.ravel() >= self.mask_threshold)[0])
            else:
                return (arr[mask.ravel() >= self.mask_threshold, :],
                        np.where(mask.ravel() >= self.mask_threshold)[0])
        else:
            if remove_borders:
                borders = self.rescale_add_borders(
                    np.ones((self.original_height, self.original_width)))
                return (arr[borders.ravel() == 1, :],
                        np.where(borders.ravel() >= self.mask_threshold)[0])
            else:
                return arr, np.where(arr.ravel() >= -1)[0]

    @staticmethod
    def repeat_pixels(img, n_times):
        img = img[np.repeat(np.arange(img.shape[0]), n_times), :]
        img = img[:, np.repeat(np.arange(img.shape[1]), n_times)]
        return img

    @staticmethod
    def local_equalize_hist(img, plot=False):
        clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
        im_equalized = clahe.apply(img)
        if plot:
            img = np.asarray(img)
            plt.figure()
            plt.subplot(2, 2, 1)
            plt.imshow(img, cmap='gray')
            plt.title('Original Image')
            plt.axis('off')
            plt.subplot(2, 2, 2)
            bins = np.arange(np.max(img) + 2) - 0.5
            plt.hist(img.ravel(), bins=np.arange(np.max(img) + 2) - 0.5, ec='white')
            plt.xlim(bins[0], bins[-1])
            plt.title('Histogram')
            plt.subplot(2, 2, 3)
            plt.imshow(im_equalized, cmap='gray')
            plt.title('Normalized Image')
            plt.axis('off')
            plt.subplot(2, 2, 4)
            bins = np.arange(np.max(im_equalized) + 2) - 0.5
            plt.hist(im_equalized.ravel(), bins=np.arange(np.max(im_equalized) + 2) - 0.5, ec='white')
            plt.xlim(bins[0], bins[-1])
            plt.title('Histogram')
            plt.show()
        return im_equalized

    @staticmethod
    def equalize_hist(img, mask=None, plot=False):
        img = Image.fromarray(np.uint8(img))
        if mask is not None:
            mask = Image.fromarray(np.uint8(mask))
        im_equalized = np.asarray(ImageOps.equalize(img, mask))
        if plot:
            img = np.asarray(img)
            plt.figure()
            plt.subplot(2, 2, 1)
            plt.imshow(img, cmap='gray')
            plt.title('Original Image')
            plt.axis('off')
            plt.subplot(2, 2, 2)
            bins = np.arange(np.max(img) + 2) - 0.5
            plt.hist(img.ravel(), bins=np.arange(np.max(img) + 2) - 0.5, ec='white')
            plt.xlim(bins[0], bins[-1])
            plt.title('Histogram')
            plt.subplot(2, 2, 3)
            plt.imshow(im_equalized, cmap='gray')
            plt.title('Normalized Image')
            plt.axis('off')
            plt.subplot(2, 2, 4)
            bins = np.arange(np.max(im_equalized) + 2) - 0.5
            plt.hist(im_equalized.ravel(), bins=np.arange(np.max(im_equalized) + 2) - 0.5, ec='white')
            plt.xlim(bins[0], bins[-1])
            plt.title('Histogram')
            plt.show()
        return im_equalized

    @staticmethod
    def median_noise_reduction(img, plot=False):
        img = Image.fromarray(np.uint8(img))
        img_filtered = np.asarray(img.filter(ImageFilter.MedianFilter(size=3)))
        if plot:
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(img, cmap='gray')
            plt.title('Original Image')
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(img_filtered, cmap='gray')
            plt.title('Filtered Image')
            plt.axis('off')
            plt.show()
        return img_filtered

    @staticmethod
    def noise_reduction(img, plot=False):
        img_filtered = cv2.fastNlMeansDenoising(img, None, 1.7, 7, 21)
        if plot:
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(img, cmap='gray')
            plt.title('Original Image')
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(img_filtered, cmap='gray')
            plt.title('Filtered Image')
            plt.axis('off')
            plt.show()
        return img_filtered

    @staticmethod
    def black_hat_filter(img, mask, plot=False):
        kernel = np.ones((3, 3), np.uint8)
        black_hat_img = cv2.morphologyEx(img,
                                         # cv2.MORPH_TOPHAT,
                                         cv2.MORPH_BLACKHAT,
                                         kernel)
        black_hat_img = Image.fromarray(np.uint8(black_hat_img))
        if mask is not None:
            mask = Image.fromarray(np.uint8(mask))
        black_hat_img = np.asarray(ImageOps.equalize(black_hat_img, mask))
        if plot:
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(img, cmap='gray')
            plt.title('Original Image')
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(black_hat_img, cmap='gray')
            plt.title('Filtered Image')
            plt.axis('off')
            plt.show()
        return black_hat_img

    def get_pyramid_dataset(self, path, label_path=None, mask_path=None, train_set=False, plot=False):
        """
        Returns the pyramid LBP values for each pixel of the given image
        :param train_set: Boolean to determine whether the dataset must be balanced
        :param path: Image path
        :param label_path: Path of the label
        :param mask_path: Path of the mask
        :param plot: Boolean to determine whether to plot the results
        :return: DataFrame of the features calculated from the image
        """
        img = Preprocess.read_img(path)
        img, mask = self.filter_by_mask(img, mask_path)
        img = self.rescale_add_borders(img)
        lbp_matrix = np.zeros((self.height * self.width, 6))
        for i in 2 ** np.arange(6):
            img_resized = Preprocess.rescale(img.copy(), (self.width // i, self.height // i))
            img_lbp = self.apply_lbp(img_resized, plot=plot)
            img_lbp = Preprocess.repeat_pixels(img_lbp, i)
            lbp_matrix[:, int(np.log2(i))] = img_lbp.ravel()
        lbp_matrix, _ = self.remove_mask_data(lbp_matrix, mask, remove_borders=True)
        if label_path is not None:
            label = self.get_label(label_path).reshape(-1, 1)
            label, _ = self.remove_mask_data(label, mask)
            lbp_matrix = np.concatenate((lbp_matrix, label), axis=1)
            if plot:
                self.plot_preprocess_with_label(lbp_matrix[:, 0], lbp_matrix[:, -1], mask)
            if train_set:
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

    def get_dataset_by_scale(self, img, mask, label, plot, train_set, i):
        img_resized = Preprocess.rescale(img, (self.width // i, self.height // i))
        img_lbp = self.apply_lbp(img_resized, plot=plot)
        if label is not None:
            label_resized = Preprocess.rescale(label.copy(), (self.width // i, self.height // i))
            if plot:
                self.plot_preprocess_with_label(img_lbp.reshape(-1, 1), label_resized.reshape(-1, 1), None, i=i)
            df = pd.DataFrame(np.concatenate((img_lbp.reshape(-1, 1), label_resized.reshape(-1, 1)), axis=1))
            if train_set:
                mask_resized = Preprocess.rescale(mask, (self.width // i, self.height // i))
                _, selected_indexes = self.remove_mask_data(np.array(df), mask_resized, remove_borders=False)
                aux = np.random.choice(np.where(df.iloc[selected_indexes, -1] == 0)[0],
                                       size=sum(df.iloc[selected_indexes, -1] == 1),
                                       replace=False)
                aux = np.concatenate((aux, np.where(df.iloc[selected_indexes, -1] == 1)[0]))
                selected_indexes = selected_indexes[np.sort(aux)]
                df = df.iloc[selected_indexes, :]
            return df
        else:
            return pd.DataFrame(img_lbp.reshape(-1, 1))

    def get_datasets_by_scale(self, path, label_path=None, mask_path=None, plot=False, train_set=True):
        img = Preprocess.read_img(path)
        img, mask = self.filter_by_mask(img, mask_path)
        img = Preprocess.noise_reduction(img, plot)  # TODO: add to pyramid dataset
        img = Preprocess.local_equalize_hist(img, plot)  # TODO: add to pyramid dataset
        img = Preprocess.noise_reduction(img, plot)  # TODO: add to pyramid dataset
        img = Preprocess.median_noise_reduction(img, plot)  # TODO: add to pyramid dataset
        # img = Preprocess.black_hat_filter(img, mask, plot)  # TODO: add to pyramid dataset
        img = self.rescale_add_borders(img)
        mask_and_borders = self.rescale_add_borders(mask)
        if label_path is not None:
            label = self.get_label(label_path)
            label = self.rescale_add_borders(label)
        else:
            label = None
        dfs = [self.get_dataset_by_scale(img.copy(), mask_and_borders.copy(), label, plot, train_set, i)
               for i in 2 ** np.arange(6)]
        if train_set:
            selected_indexes = None
        else:
            _, selected_indexes = self.remove_mask_data(np.array(dfs[0]), mask, remove_borders=True)
        return {'datasets': dfs, 'mask': selected_indexes}

    def get_label(self, path):
        img = Preprocess.read_img(path)
        img[img < self.label_threshold] = 0
        img[img > self.label_threshold] = 1
        return img

    def plot_preprocess_with_label(self, img, label, mask, i=1):
        def array_to_mat(arr, mask_mat):
            mat = np.copy(mask_mat)
            mat[mat < self.mask_threshold] = 0
            mat[mat >= self.mask_threshold] = arr.ravel()
            return mat

        if mask is None:
            mask = np.ones((self.height // i, self.width // i)) * self.mask_threshold
        img = array_to_mat(img, mask)
        img = (img * (255 / np.max(img))).astype(int)
        label = array_to_mat(label, mask)
        img = np.asarray(Image.fromarray(np.uint8(img)).convert('RGB')).copy()
        img[label == 1] = [255, 0, 0]
        im = Image.fromarray(np.uint8(img))
        im.show()
