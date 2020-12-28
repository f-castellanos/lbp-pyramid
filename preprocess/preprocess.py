from .lbp import lbp
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import OneHotEncoder
import pickle


class Preprocess:
    def __init__(self, height, width, balance=False, lbp_radius=1,
                 lbp_method='default', mask_threshold=100, label_threshold=30):
        self.height = height
        self.width = width
        self.balance = balance
        self.original_height = 584
        self.original_width = 565
        self.mask_threshold = mask_threshold
        self.label_threshold = label_threshold
        self.lbp_radius = lbp_radius
        self.lbp_method = lbp_method

    ##  Basic Operations
    @staticmethod
    def read_img(path):
        """
        Reads a image given the path
        :param path: Image path
        :return: Numpy array containing the information of the image
        """
        img = np.asarray(Image.open(path).convert('L'))
        return img.copy()

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
    def repeat_pixels(img, n_times):
        img = img[np.repeat(np.arange(img.shape[0]), n_times), :]
        img = img[:, np.repeat(np.arange(img.shape[1]), n_times)]
        return img

    @staticmethod
    def gray_to_array(img):
        return img.ravel()

    ## Mask Operations
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

    ##  LBP
    def apply_lbp(self, img, plot=False):
        """
        Returns the LBP values for the given image
        :param img: Numpy array containing the information of the image.
        :param plot: Boolean to determine whether to plot the results.
        :return: Numpy array containing the information of the LBP image.
        """
        img = lbp(img, r=self.lbp_radius, method=self.lbp_method, plot=plot)
        return img

    ## Image transformations
    @staticmethod
    def local_equalize_hist(img, plot=False):
        cla_he = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
        im_equalized = cla_he.apply(img)
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
    def noise_reduction(img, plot=False, d=3, sigma_color=3, sigma_space=9):
        """
        Bilateral Filter: Highly effective in noise removal while keeping edges sharp
        :param sigma_space: A larger value of the parameter means that farther pixels will influence each other as long
                            as their colors are close enough
        :param d: Diameter of each pixel neighborhood that is used during filtering.
        :param img: Input image.
        :param plot: Boolean to determine whether to plot the results.
        :param sigma_color: A larger value of the parameter means that farther colors within the pixel neighborhood will
                            be mixed together.
        :return: Output image.
        """
        img_filtered = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
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
    def img_processing(img, plot=False):
        img = Preprocess.noise_reduction(img, plot)
        img = Preprocess.local_equalize_hist(img, plot)
        img = Preprocess.noise_reduction(img, plot)
        return img

    ## Gold Standard
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

    ## Encoding
    def one_hot_encode(self, df, multiple_scale=True):
        encoder = OneHotEncoder()
        df = df.applymap(str)
        encoder.fit(df)
        if multiple_scale:
            columns = encoder.get_feature_names(df.columns)
            df = pd.DataFrame(encoder.transform(df).toarray(), columns=columns).applymap(int)
            with open('preprocess/' + self.lbp_method + '_column_list.pkl', 'rb') as f_columns:
                column_list = pickle.load(f_columns)
        else:
            df = pd.DataFrame(encoder.transform(df).toarray()).applymap(int)
            df.columns = np.array(df.columns).astype(str)
            lbp_len = {'default': 256, 'riu': 36, 'riu2': 10}
            column_list = list(np.arange(lbp_len[self.lbp_method]).astype(str))
        missing_columns = np.setdiff1d(column_list, list(df.columns))
        if len(missing_columns) > 0:
            for column in missing_columns:
                df[column] = np.zeros(df.shape[0]).astype(int)
        return df.loc[:, column_list]

    ## Single model dataset constructor
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
        img = Preprocess.img_processing(img, plot)
        img = self.rescale_add_borders(img)
        lbp_matrix = np.zeros((self.height * self.width, 6))
        for i in 2 ** np.arange(6):
            img_resized = Preprocess.rescale(img.copy(), (self.width // i, self.height // i))
            img_lbp = self.apply_lbp(img_resized, plot=plot)
            img_lbp = Preprocess.repeat_pixels(img_lbp, i)
            lbp_matrix[:, int(np.log2(i))] = img_lbp.ravel()
        #  Original image
        lbp_matrix = np.concatenate((
                img.ravel().reshape(-1, 1),
                lbp_matrix,
            ),
            axis=1
        )
        lbp_matrix, _ = self.remove_mask_data(lbp_matrix, mask, remove_borders=True)
        if label_path is not None:
            label = self.get_label(label_path).reshape(-1, 1)
            label, _ = self.remove_mask_data(label, mask)
            lbp_matrix = np.concatenate((lbp_matrix, label), axis=1)
            if plot:
                self.plot_preprocess_with_label(lbp_matrix[:, 0], lbp_matrix[:, -1], mask)
            if train_set and self.balance is True and sum(lbp_matrix[:, -1] == 1) > 0:
                random_sample = np.random.choice(np.where(lbp_matrix[:, -1] == 0)[0],
                                                 size=sum(lbp_matrix[:, -1] == 1),
                                                 replace=False)
                sample = np.sort(
                    np.concatenate((random_sample, np.where(lbp_matrix[:, -1] == 1)[0]))
                )
                lbp_matrix = lbp_matrix[sample, :]
            df = pd.DataFrame(lbp_matrix, columns=['Original', '1:1', '1:2', '1:4', '1:8', '1:16', '1:32', 'label'],
                              dtype='uint8')
            df = pd.concat((df.loc[:, 'Original'], self.one_hot_encode(df.iloc[:, 1:-1].copy()), df.loc[:, 'label']),
                           axis=1)
        else:
            df = self.one_hot_encode(
                pd.DataFrame(lbp_matrix, columns=['1:1', '1:2', '1:4', '1:8', '1:16', '1:32'], dtype='uint8')
            )
            df = pd.concat((df.loc[:, 'Original'], self.one_hot_encode(df.iloc[:, 1].copy())), axis=1)
        return df

    ## Multiple models dataset constructor
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
                if self.balance is True and sum(df.iloc[selected_indexes, -1] == 1) > 0:
                    aux = np.random.choice(np.where(df.iloc[selected_indexes, -1] == 0)[0],
                                           size=sum(df.iloc[selected_indexes, -1] == 1),
                                           replace=False)
                    aux = np.concatenate((aux, np.where(df.iloc[selected_indexes, -1] == 1)[0]))
                    selected_indexes = selected_indexes[np.sort(aux)]
                df = pd.concat((
                    pd.Series(img.ravel()),
                    df
                ), axis=1)
                df = df.iloc[selected_indexes, :]
            return pd.concat(
                (
                    df.iloc[:, 0].reset_index(drop=True),
                    self.one_hot_encode(df.iloc[:, 1].to_frame().copy(), multiple_scale=False).reset_index(drop=True),
                    df.iloc[:, -1].reset_index(drop=True)
                ),
                axis=1
            )
        else:
            return pd.concat(
                (
                    pd.Series(img.ravel()),
                    self.one_hot_encode(pd.DataFrame(img_lbp.reshape(-1, 1)), multiple_scale=False)
                ),
                axis=1
            )

    def get_datasets_by_scale(self, path, label_path=None, mask_path=None, plot=False, train_set=True):
        img = Preprocess.read_img(path)
        img, mask = self.filter_by_mask(img, mask_path)
        img = Preprocess.img_processing(img, plot)
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
