import _pickle as pickle
import bz2
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from joblib import Parallel, delayed, parallel_backend
from sklearn.preprocessing import OneHotEncoder

import PARAMETERS
if 'J_NOTEBOOK' in os.environ and os.environ['J_NOTEBOOK'] == '1':
    from .lbp_scikit import lbp
else:
    from lbp_scikit import lbp


class ParameterError(Exception):
    pass


VALID_PARAMETERS = {
    # 'LBP_METHOD': ['var'],
    'LBP_METHOD': ['riu', 'var', 'default'],
    # 'LBP_METHOD': ['default', 'riu', 'riu2', 'nriuniform', 'var'],
    'METHOD': ['get_pyramid_dataset'],
    # 'METHOD': ['get_pyramid_dataset', 'get_datasets_by_scale'],
    'INTERPOLATION_ALGORITHM': ['lanczos'],
    # 'INTERPOLATION_ALGORITHM': ['nearest', 'lanczos', 'bicubic'],
    'BALANCE': [False],
    # 'BALANCE': [False, True],
    'N_SCALES': [5],
    # 'N_SCALES': list(range(1, 7)),
    'GRAY_INTENSITY': [True],
    # 'GRAY_INTENSITY': [True, False],
    'X2SCALE': [True],
    # 'X2SCALE': [False, True],
}


PARENT_PATH = str(Path(os.path.dirname(os.path.abspath(__file__))).parent.parent)
PREPROCESS_PARAMS = {
    'DRIVE_GB': np.array([23,  7, 64, 65, 40, 54, 38, 42]),
    'DRIVE_W': np.array([35,   7,  24, 166,  45,   4,  86,  35]),
    'DRIVE_LBP': np.array([59, 43,  0,  9, 96, 17, 16, 64]),
    'DRIVE_LBP_GB': np.array([20,  8, 71, 61, 46, 68, 30, 36]),
    'DRIVE_LBP_G': np.array([41,  6, 46, 60, 32, 32, 42, 47]),
    'DRIVE_TEST_GB': np.array([23,  7, 64, 65, 40, 54, 38, 42]),
    'DRIVE_TEST_W': np.array([35,   7,  24, 166,  45,   4,  86,  35]),
    'DRIVE_TEST_LBP': np.array([59, 43,  0,  9, 96, 17, 16, 64]),
    'DRIVE_TEST_LBP_GB': np.array([20,  8, 71, 61, 46, 68, 30, 36]),
    'DRIVE_TEST_LBP_G': np.array([41,  6, 46, 60, 32, 32, 42, 47]),
    'STARE_LBP_GB': np.array([60, 11, 74, 52, 29, 32, 29, 28]),
    'STARE_LBP_G': np.array([58, 10, 74, 51, 29, 30, 28, 31]),
    'STARE_LBP_G_CV': np.array([53,  1, 24, 66, 65, 92, 86,  2]),
    'STARE_LBP_G_FOLD_0_of_5': np.array([47,  6, 45, 58, 35, 38, 51, 41]),
    'STARE_LBP_G_FOLD_1_of_5': np.array([41,  7, 46, 57, 37, 39, 46, 34]),
    'STARE_LBP_G_FOLD_2_of_5': np.array([55, 11, 38, 39, 46, 23, 15, 60]),
    'STARE_LBP_G_FOLD_3_of_5': np.array([53, 11, 26, 56, 44, 41, 43, 43]),
    'STARE_LBP_G_FOLD_4_of_5': np.array([41, 11, 85, 60, 54, 72,  6, 34]),
    'CHASE_LBP_G': np.array([25,  6, 43, 76, 35, 38, 32, 44]),
}


class Preprocess:
    def __init__(self, height, width, balance=False, lbp_radius=1, fold=None,
                 lbp_method='default', mask_threshold=100, label_threshold=30):
        self.height = height
        self.width = width
        self.balance = balance
        self.original_height = {'DRIVE': 584, 'DRIVE_TEST': 584, 'CHASE': 960, 'STARE': 605}[PARAMETERS.DATASET]
        self.original_width = {'DRIVE': 565, 'DRIVE_TEST': 565, 'CHASE': 999, 'STARE': 700}[PARAMETERS.DATASET]
        self.mask_threshold = mask_threshold
        self.label_threshold = label_threshold
        self.lbp_radius = lbp_radius
        self.lbp_method = lbp_method
        self.fold = fold
        self.img_order = {
            'DRIVE': np.arange(20),
            'DRIVE_TEST': np.arange(20),
            'CHASE': np.arange(28),
            'STARE': np.array([3, 16, 6, 10, 2, 14, 4, 17, 7, 1, 13, 0, 19, 18, 9, 15, 8, 12, 11, 5]) if fold is not None else np.arange(20)  # noqa
        }[PARAMETERS.DATASET]
        self.fold_size = None
        if fold is not None:
            dataset_size = {'DRIVE': 20, 'DRIVE_TEST': 20, 'CHASE': 28, 'STARE': 20}[PARAMETERS.DATASET]
            self.fold_size = dataset_size - int(dataset_size - dataset_size/PARAMETERS.FOLDS)
            self.img_order = np.roll(self.img_order, np.arange(0, dataset_size, self.fold_size)[fold])
        self.parameters_verification()
        self.training_path = None
        self.images_path = None
        self.masks_path = None
        self.preprocessed_path = None
        self.original_preprocessed_path = None

    def compute_preprocessing(self, filenames, mask_filenames, main_path):
        self.training_path = main_path
        self.images_path = f"{self.training_path}images"
        self.masks_path = f"{self.training_path}mask"
        if self.fold is not None:
            self.training_path += f'fold_{self.fold}_of_{PARAMETERS.FOLDS}/'
        if PARAMETERS.CONVOLUTION is None:
            self.preprocessed_path = f"{self.training_path}preprocessed"
            if PARAMETERS.CHANNEL is not None:
                channels_map = {0: 'red', 1: 'green', 2: 'blue'}
                self.preprocessed_path += f"_{channels_map[PARAMETERS.CHANNEL]}_channel"
            if PARAMETERS.PREPROCESS_OPTIMIZATION:
                self.preprocessed_path += f'_optimized' + {
                    "default": "", "gb": "_gb", "w": "_w", "lbp": "_lbp", 'lbp_gb': "_lbp_gb", 'lbp_g': "_lbp_g",
                    "lbp_g_cv": "_lbp_g_cv", "lbp_g_fold": "_lbp_g"
                }[PARAMETERS.PREPROCESS_TYPE]
        else:
            PARAMETERS.CONV_PATH = PARAMETERS.update_convolution_path(PARAMETERS)
            self.preprocessed_path = f"{self.training_path}preprocessed_convolutions/{PARAMETERS.CONV_PATH}"
            self.original_preprocessed_path = f"{self.training_path}preprocessed"
        # for algorithm in VALID_PARAMETERS['INTERPOLATION_ALGORITHM']:
        with parallel_backend('multiprocessing', n_jobs=PARAMETERS.N_JOBS):
            _ = Parallel()(
                delayed(self.compute_for_interpolation_algorithm)(algorithm, filenames, mask_filenames)
                for algorithm in VALID_PARAMETERS['INTERPOLATION_ALGORITHM']
            )
        # self.compute_for_interpolation_algorithm(algorithm, filenames, mask_filenames)

    def compute_for_interpolation_algorithm(self, algorithm, filenames, mask_filenames):
        # PARAMETERS.INTERPOLATION_ALGORITHM = algorithm
        algorithm_path = f"{self.preprocessed_path}/{algorithm}/original"
        if not os.path.exists(algorithm_path):
            os.makedirs(algorithm_path)
        for filename, mask_filename in zip(filenames, mask_filenames):
            if len(list(Path(algorithm_path).glob(f"{filename.split('.')[0]}*"))) == 7:
                continue
            if PARAMETERS.CONVOLUTION is None:
                img = Preprocess.read_img(f"{self.images_path}/{filename}")
                img, mask = self.filter_by_mask(img, f"{self.masks_path}/{mask_filename}")
                # if PARAMETERS.CHANNEL is None:
                params = None
                if PARAMETERS.PREPROCESS_OPTIMIZATION:
                    params = PREPROCESS_PARAMS[PARAMETERS.DATASET + {
                        "default": "", "gb": "_GB", "w": "_W", "lbp": "_LBP", 'lbp_gb': "_LBP_GB", 'lbp_g': "_LBP_G",
                        "lbp_g_cv": "_LBP_G_CV", "lbp_g_fold": f"_LBP_G_FOLD_{self.fold}_of_{PARAMETERS.FOLDS}",
                    }[PARAMETERS.PREPROCESS_TYPE]]
                img = Preprocess.img_processing(img, PARAMETERS.PLOT, params=params)
                img = self.rescale_add_borders(img)
                for i in float(2) ** np.arange(-1, 6):
                    img_resized = Preprocess.rescale(
                        img.copy(), (self.width // i, self.height // i), algorithm=algorithm)
                    im = Image.fromarray(img_resized)
                    im.save(f"{algorithm_path}/{filename.split('.')[0]}_{i}.jpeg")
            else:
                for i in float(2) ** np.arange(-1, 7):
                # for i in float(2) ** np.arange(-1, 6):
                    if PARAMETERS.CONV_PREPROCESSING:
                        img = Preprocess.read_img(f"{self.original_preprocessed_path}/{algorithm}/original/{filename.split('.')[0]}_{i}.jpeg")  # noqa
                    else:
                        img = Preprocess.read_img(f"{self.images_path}/{filename}")
                        img = self.rescale_add_borders(img)
                        img = Preprocess.rescale(
                            img.copy(), (self.width // i, self.height // i), algorithm=algorithm)
                    img = cv2.filter2D(img, -1, PARAMETERS.CONVOLUTION)
                    img = np.round((img/np.max(img)) * 255).astype(np.int8)
                    with bz2.BZ2File(f"{algorithm_path}/{filename.split('.')[0]}_{i}.pkl", 'wb') as f:
                        pickle.dump(img, f)

        self.compute_lbp(algorithm)

    def compute_lbp(self, algorithm):
        algorithm_path = f"{self.preprocessed_path}/{algorithm}"
        original_path = f"{algorithm_path}/original"
        filenames = list(Path(original_path).glob("*"))
        for lbp_operator in VALID_PARAMETERS['LBP_METHOD']:
            lbp_path = f"{algorithm_path}/lbp/{lbp_operator}"
            if not os.path.exists(lbp_path):
                os.makedirs(lbp_path)
            for filename in filenames:
                filename = str(filename)
                new_filename = filename.split('/')[-1].replace('.jpeg', '.pkl')
                if not os.path.isfile(f"{lbp_path}/{new_filename}"):
                    if PARAMETERS.CONVOLUTION is None:
                        img = Preprocess.read_img(filename)
                        img_lbp = self.apply_lbp(img, method=lbp_operator, plot=PARAMETERS.PLOT)
                    else:
                        with bz2.BZ2File(filename.replace('.jpeg', '.pkl'), 'rb') as f:
                            img = pickle.load(f)
                        img_lbp = self.apply_lbp(img, method=lbp_operator, plot=PARAMETERS.PLOT)
                    with bz2.BZ2File(f"{lbp_path}/{new_filename}", 'wb') as f:
                        pickle.dump(img_lbp, f)
                # Processing of radius 2-4
                if PARAMETERS.MULTI_RADIUS:
                    for r in range(2, 6):
                    # for r in range(2, 5):
                        # riu with r > 2 freezes the process
                        if (not os.path.isfile(f"{lbp_path}_{r}/{new_filename}")) and lbp_operator != 'riu':
                            if not os.path.exists(f"{lbp_path}_{r}"):
                                os.makedirs(f"{lbp_path}_{r}")
                            if PARAMETERS.CONVOLUTION is None:
                                img = Preprocess.read_img(filename)
                                img_lbp = self.apply_lbp(img, method=lbp_operator, plot=PARAMETERS.PLOT, r=r)
                            else:
                                with bz2.BZ2File(filename.replace('.jpeg', '.pkl'), 'rb') as f:
                                    img = pickle.load(f)
                                img_lbp = self.apply_lbp(img, method=lbp_operator, plot=PARAMETERS.PLOT, r=r)
                            with bz2.BZ2File(f"{lbp_path}_{r}/{new_filename}", 'wb') as f:
                                pickle.dump(img_lbp, f)
                if PARAMETERS.RADIUS > 1:
                    for r in range(2, 7):
                        if not os.path.isfile(f"{lbp_path}_{r}/{new_filename}") and lbp_operator == PARAMETERS.LBP_METHOD:
                            if not os.path.exists(f"{lbp_path}_{r}"):
                                os.makedirs(f"{lbp_path}_{r}")
                            img = Preprocess.read_img(filename)
                            img_lbp = self.apply_lbp(img, method=lbp_operator, plot=PARAMETERS.PLOT, r=r)
                            with bz2.BZ2File(f"{lbp_path}_{r}/{new_filename}", 'wb') as f:
                                pickle.dump(img_lbp, f)

    @staticmethod
    def parameters_verification():
        """
        Verification of the given execution parameters
        :return: None
        """
        if 'SKIP_VALIDATION' not in os.environ or os.environ['SKIP_VALIDATION'] != 'True':
            for k, v in VALID_PARAMETERS.items():
                pass
                # if getattr(PARAMETERS, k) not in v:
                #     raise ParameterError(f"{getattr(PARAMETERS, k)} is not correctly defined")

    @staticmethod
    def read_img(path):
        """
        Reads a image given the path
        :param x2_enabled: Enables x2 rescaling
        :param path: Image path
        :return: Numpy array containing the information of the image
        """
        if PARAMETERS.CHANNEL is None:
            img = np.asarray(Image.open(path).convert('L'))
        else:
            img = np.asarray(Image.open(path).convert('RGB'))[:, :, PARAMETERS.CHANNEL]
        return img.copy()

    @staticmethod
    def rescale(img, dim, algorithm=None):
        """
        Resizes the image to the given dimensions.
        PIL.Image.LANCZOS – Calculate the output pixel value using a high-quality Lanczos filter on
        all pixels that may contribute to the output value.
        https://pillow.readthedocs.io/en/stable/handbook/concepts.html
        :param img: original image
        :param dim: (width, height)
        :return: image resized
        """
        resample_map = {
            'lanczos': Image.LANCZOS,
            'nearest': Image.NEAREST,
            'bicubic': Image.BICUBIC
        }
        im = Image.fromarray(np.uint8(img))
        if algorithm is None:
            algorithm = PARAMETERS.INTERPOLATION_ALGORITHM
        if img.shape != dim:
            return np.asarray(im.resize((int(dim[0]), int(dim[1])), resample=resample_map[algorithm]))
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
    def undo_repeat_pixels(img):
        """

        :param img:
        :return:

        Example:

        img = array([[ 0,  1,  2,  3,  4,  5],
                     [ 6,  7,  8,  9, 10, 11],
                     [12, 13, 14, 15, 16, 17],
                     [18, 19, 20, 21, 22, 23]])

        i_indexes = array([[0, 0, 1, 1, 2, 2],
                           [0, 0, 1, 1, 2, 2],
                           [3, 3, 4, 4, 5, 5],
                           [3, 3, 4, 4, 5, 5]])

        j_indexes = array([[0, 1, 0, 1, 0, 1],
                           [2, 3, 2, 3, 2, 3],
                           [0, 1, 0, 1, 0, 1],
                           [2, 3, 2, 3, 2, 3]])

        Then, new columns for pixel 0 are (0, 1, 6, 7)
        """
        i_indexes = Preprocess.repeat_pixels(
            np.arange(img.shape[0]//2*img.shape[1]//2).reshape((img.shape[0]//2, img.shape[1]//2)), 2)

        x = np.array([[0, 1], [2, 3]])
        j_indexes = np.tile(x, (img.shape[0]//2, img.shape[1]//2))

        reshaped_img = np.zeros((img.shape[0]*img.shape[1] // 4, 4))
        # reshaped_img = np.zeros((img.shape[0]*img.shape[1] // 4, 4), dtype='uint8')
        for v, i, j in zip(img.ravel(), i_indexes.ravel(), j_indexes.ravel()):
            reshaped_img[i, j] = v

        return reshaped_img

    @staticmethod
    def gray_to_array(img):
        return img.ravel()

    # Mask Operations
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
            mask = self.read_img(mask_path)
            # img[mask < self.mask_threshold] = np.median(img[mask >= self.mask_threshold])
            img[mask < self.mask_threshold] = 0
        else:
            mask = None
        return img, mask

    #  LBP
    def apply_lbp(self, img, plot=False, method=None, r=1):
        """
        Returns the LBP values for the given image
        :param img: Numpy array containing the information of the image.
        :param plot: Boolean to determine whether to plot the results.
        :return: Numpy array containing the information of the LBP image.
        """
        # scale = self.scale/2 if PARAMETERS.X2SCALE else self.scale
        if method is None:
            method = self.lbp_method
        lbp_img = lbp(img, r=r, method=method, plot=plot)
        return np.nan_to_num(lbp_img)

    # Image transformations
    @staticmethod
    def local_equalize_hist(img, plot=False, clip_limit=3, tile_grid_size=8):
        cla_he = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
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
    def img_processing(img, plot=False, params=None):
        if params is None:
            img = Preprocess.noise_reduction(img, plot)
            img = Preprocess.local_equalize_hist(img, plot)
            img = Preprocess.noise_reduction(img, plot)
        else:
            img = Preprocess.noise_reduction(img, plot, *params[:3])
            img = Preprocess.local_equalize_hist(img, plot, *params[3:5])
            img = Preprocess.noise_reduction(img, plot, *params[5:])
        return img

    # Gold Standard
    def get_label(self, path):
        img = self.read_img(path)
        img[img < self.label_threshold] = 0
        img[img >= self.label_threshold] = 1
        # if PARAMETERS.X2SCALE:
        #     img = Preprocess.repeat_pixels(img, 2)
        return img

    def plot_preprocess_with_label(self, img, label, mask, i=1):
        def array_to_mat(arr, mask_mat):
            mat = np.copy(mask_mat)
            mat[mat < self.mask_threshold] = 0
            mat[mat >= self.mask_threshold] = arr.ravel()
            return mat

        if mask is None:
            mask = np.ones((int(self.height // i), int(self.width // i))) * self.mask_threshold
        img = array_to_mat(img, mask)
        img = (img * (255 / np.max(img))).astype(int)
        label = array_to_mat(label, mask)
        img = np.asarray(Image.fromarray(np.uint8(img)).convert('RGB')).copy()
        img[label == 1] = [255, 0, 0]
        # img[label == 0] = [0, 0, 0]
        im = Image.fromarray(np.uint8(img))
        plt.figure(figsize=(15, 11), dpi=80)
        plt.axis('off')
        plt.imshow(im)
        plt.show()
        # im.show()

    # Encoding
    def one_hot_encode(self, df, multiple_scale=True):
        original_columns = list(df.columns)
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
                if PARAMETERS.METHOD == 'get_datasets_by_scale' or column.split('_')[-2] in original_columns:
                    df[column] = np.zeros(df.shape[0]).astype(int)
                else:
                    column_list.remove(column)
        return df.loc[:, column_list]

    # Single model dataset constructor
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
        filename = path.split('/')[-1]
        path = f"{self.preprocessed_path}/{PARAMETERS.INTERPOLATION_ALGORITHM}/lbp/{PARAMETERS.LBP_METHOD}"
        if PARAMETERS.RADIUS > 1:
            path += f'_{PARAMETERS.RADIUS}'
        # elif PARAMETERS.RADIUS > 1:
        #     raise Exception
        scale_names = ['1:1', '1:2', '1:4', '1:8', '1:16', '1:32'][:(PARAMETERS.N_SCALES - int(PARAMETERS.X2SCALE))]
        if PARAMETERS.X2SCALE:
            scale_names += ['2:1_1', '2:1_2', '2:1_3', '2:1_4']
            # lbp_matrix = np.zeros((self.height * self.width, PARAMETERS.N_SCALES + 3), dtype='uint8')
            lbp_matrix = np.zeros((self.height * self.width, PARAMETERS.N_SCALES + 3))
            with bz2.BZ2File(f"{path}/{filename.split('.tif')[0]}_0.5.pkl".replace('.ppm', '').replace('.jpg', ''), 'rb') as f:
                img_lbp = pickle.load(f)
            lbp_matrix[:, -4:] = Preprocess.undo_repeat_pixels(img_lbp)
        else:
            # lbp_matrix = np.zeros((self.height * self.width, PARAMETERS.N_SCALES), dtype='uint8')
            lbp_matrix = np.zeros((self.height * self.width, PARAMETERS.N_SCALES))

        for i in 2 ** np.arange(PARAMETERS.N_SCALES - int(PARAMETERS.X2SCALE)):
            with bz2.BZ2File(f"{path}/{filename.split('.tif')[0].split('.jpg')[0].split('.ppm')[0]}_{float(i)}.pkl", 'rb') as f:
                img_lbp = pickle.load(f)
            img_lbp = Preprocess.repeat_pixels(img_lbp, i)
            lbp_matrix[:, int(np.log2(i))] = img_lbp.ravel()
        #  Original image
        img_path = f"{self.preprocessed_path}/{PARAMETERS.INTERPOLATION_ALGORITHM}/original/" \
                   f"{filename.split('.tif')[0]}_1.0.jpeg".replace('.ppm', '').replace('.jpg', '')
        if PARAMETERS.CONVOLUTION is None:
            img = Preprocess.read_img(img_path)
        else:
            with bz2.BZ2File(img_path.replace('.jpeg', '.pkl'), 'rb') as f:
                img = pickle.load(f)
        mask = self.read_img(mask_path)
        if PARAMETERS.GRAY_INTENSITY:
            lbp_matrix = np.concatenate((
                    img.copy().ravel().reshape(-1, 1),
                    lbp_matrix,
                ),
                axis=1
            )
            # Binning
            lbp_matrix[:, 0] = np.round(lbp_matrix[:, 0] / 25)
        lbp_matrix, _ = self.remove_mask_data(lbp_matrix, mask, remove_borders=True)
        # scale_names = ['1:0.5', '1:1', '1:2', '1:4', '1:8', '1:16', '1:32'][
        #               int(not PARAMETERS.X2SCALE):PARAMETERS.N_SCALES+int(not PARAMETERS.X2SCALE)]
        if label_path is not None:
            label = self.get_label(label_path).reshape(-1, 1)
            label, _ = self.remove_mask_data(label, mask)
            lbp_matrix = np.concatenate((lbp_matrix, label), axis=1)
            if plot or PARAMETERS.PLOT_LBP_LABEL:
                self.plot_preprocess_with_label(lbp_matrix[:, 0], lbp_matrix[:, -1], mask)
            if train_set and self.balance is True and sum(lbp_matrix[:, -1] == 1) > 0:
                random_sample = np.random.choice(np.where(lbp_matrix[:, -1] == 0)[0],
                                                 size=sum(lbp_matrix[:, -1] == 1),
                                                 replace=False)
                sample = np.sort(
                    np.concatenate((random_sample, np.where(lbp_matrix[:, -1] == 1)[0]))
                )
                lbp_matrix = lbp_matrix[sample, :]

            if PARAMETERS.GRAY_INTENSITY:
                df = pd.DataFrame(lbp_matrix, columns=['Original'] + scale_names + ['label'])
                # df = pd.DataFrame(lbp_matrix, columns=['Original'] + scale_names + ['label'], dtype='uint8')
                if PARAMETERS.ENCODING == 'one-hot':
                    df = pd.concat(
                        (df.loc[:, 'Original'], self.one_hot_encode(df.iloc[:, 1:-1].copy()), df.loc[:, 'label']),
                        axis=1
                    )
                elif PARAMETERS.ENCODING == 'categorical':
                    for col in df.iloc[:, :-1].columns:
                        df[col] = df[col].astype('category')
            else:
                df = pd.DataFrame(lbp_matrix, columns=scale_names + ['label'])
                # df = pd.DataFrame(lbp_matrix, columns=scale_names + ['label'], dtype='uint8')
                if PARAMETERS.ENCODING == 'one-hot':
                    df = pd.concat(
                        (self.one_hot_encode(df.iloc[:, :-1].copy()), df.loc[:, 'label']),
                        axis=1
                    )
                elif PARAMETERS.ENCODING == 'categorical':
                    for col in df.iloc[:, :-1].columns:
                        df[col] = df[col].astype('category')
        else:
            if PARAMETERS.GRAY_INTENSITY:
                df = pd.DataFrame(lbp_matrix, columns=['Original'] + scale_names)
                # df = pd.DataFrame(lbp_matrix, columns=['Original'] + scale_names, dtype='uint8')
                if PARAMETERS.ENCODING == 'one-hot':
                    df = pd.concat(
                        (df.loc[:, 'Original'], self.one_hot_encode(df.iloc[:, 1:].copy())),
                        axis=1
                    )
                elif PARAMETERS.ENCODING == 'categorical':
                    for col in df.iloc[:, :-1].columns:
                        df[col] = df[col].astype('category')
            else:
                df = pd.DataFrame(lbp_matrix, columns=scale_names)
                # df = pd.DataFrame(lbp_matrix, columns=scale_names, dtype='uint8')
                if PARAMETERS.ENCODING == 'one-hot':
                    df = self.one_hot_encode(df.iloc[:, 1:-1].copy())
                elif PARAMETERS.ENCODING == 'categorical':
                    for col in df.iloc[:, :-1].columns:
                        df[col] = df[col].astype('category')
        # if df.shape[1] < 1:
        #     a = 0
        return df

    # Multiple models dataset constructor
    def get_dataset_by_scale(self, img, mask, label, plot, train_set, i):
        img_resized = Preprocess.rescale(img, (self.width // i, self.height // i))
        # self.scale = i
        img_lbp = self.apply_lbp(img_resized, plot=plot)
        if label is not None:
            label_resized = Preprocess.rescale(label.copy(), (self.width // i, self.height // i))
            if plot or PARAMETERS.PLOT_LBP_LABEL:
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
        # self.path = path
        img = self.read_img(path)
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
               for i in 2 ** np.arange(PARAMETERS.N_SCALES)]
        if train_set:
            selected_indexes = None
        else:
            _, selected_indexes = self.remove_mask_data(np.array(dfs[0]), mask, remove_borders=True)
        return {'datasets': dfs, 'mask': selected_indexes}

# path = '/home/fer/Nextcloud/Master-IA/TFM/dataset/training/1st_manual/21_manual1.gif'
# img = np.asarray(Image.open(path).convert('L'))
# plt.figure()
# plt.imshow(img, cmap='gray')
# plt.show()
# im = Image.fromarray(np.uint8(img))
# # img = np.asarray(im.resize((int(584 * 2), int(565 *2)), resample=Image.BICUBIC))
# img = np.asarray(im.resize((int(584 / 4), int(565 /4)), resample=Image.NEAREST))
# plt.figure()
# plt.imshow(img, cmap='gray')
# plt.show()
