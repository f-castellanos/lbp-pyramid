from .lbp import lbp
import pandas as pd
import numpy as np
import PIL


class Preprocess:
    def __init__(self, height, width, lbp_radius=1, lbp_method='default', mask_threshold=100, label_threshold=30):
        HEIGHT = height
        WIDTH = width
        ORIGINAL_HEIGHT = 584
        ORIGINAL_WIDTH = 565
        MASK_THRESHOLD = mask_threshold
        LABEL_THRESHOLD = label_threshold
        LBP_RADIUS = lbp_radius
        LBP_METHOD = lbp_method

    @staticmethod
    def read_img(path):
        """
        Reads a image given the path
        :param path: Image path
        :return: Numpy array containing the information of the image
        """
        img = np.asarray(PIL.Image.open(path).convert('L'))
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
            # img[mask < MASK_THRESHOLD] = np.median(img[mask >= MASK_THRESHOLD])
            img[mask < MASK_THRESHOLD] = 0
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
        im = PIL.Image.fromarray(np.uint8(img))
        if img.shape != dim:
            return np.asarray(im.resize(dim, resample=PIL.Image.LANCZOS))
        else:
            return img

    def crop(self, img, dim=(512, 512)):
        crop_height = (HEIGHT - dim[0]) / 2
        if crop_height == int(crop_height):
            crop_bottom = int(crop_height)
            crop_top = HEIGHT - int(crop_height)
        else:
            crop_bottom = int(crop_height)
            crop_top = HEIGHT - (int(crop_height) + 1)
        crop_width = (WIDTH - dim[1]) / 2
        if crop_width == int(crop_width):
            crop_left = int(crop_width)
            crop_right = WIDTH - int(crop_width)
        else:
            crop_left = int(crop_width)
            crop_right = WIDTH - (int(crop_width) + 1)
        return img[crop_bottom:crop_top, crop_left:crop_right, :]

    def rescale_add_borders(self, img):
        """
        Adds dummy pixels to the given image to rescale the image to the desired size.
        :param img: Numpy array containing the information of the image.
        :return: Numpy array containing the information of the resized image.
        """
        height, width = img.shape
        v_add = np.zeros((HEIGHT - height, width))
        h_add = np.zeros((HEIGHT, WIDTH - width))
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
        img = lbp(img, r=LBP_RADIUS, method=LBP_METHOD, plot=plot)
        return img

    def remove_mask_data(self, arr, mask, remove_borders=False):
        if mask is not None:
            if remove_borders:
                mask_and_borders = self.rescale_add_borders(mask)
                return (arr[mask_and_borders.ravel() >= MASK_THRESHOLD, :],
                        np.where(mask_and_borders.ravel() >= MASK_THRESHOLD)[0])
            else:
                return (arr[mask.ravel() >= MASK_THRESHOLD, :],
                        np.where(mask.ravel() >= MASK_THRESHOLD)[0])
        else:
            if remove_borders:
                borders = self.rescale_add_borders(
                    np.ones((ORIGINAL_HEIGHT, ORIGINAL_WIDTH)))
                return (arr[borders.ravel() == 1, :],
                        np.where(borders.ravel() >= MASK_THRESHOLD)[0])
            else:
                return arr, np.where(arr.ravel() >= -1)[0]

    @staticmethod
    def repeat_pixels(img, n_times):
        img = img[np.repeat(np.arange(img.shape[0]), n_times), :]
        img = img[:, np.repeat(np.arange(img.shape[1]), n_times)]
        return img

    def get_pyramid_dataset(self, path, label_path=None, mask_path=None, balance=False, plot=False):
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
        img, mask = self.filter_by_mask(img, mask_path)
        img = self.rescale_add_borders(img)
        lbp_matrix = np.zeros((HEIGHT * WIDTH, 6))
        for i in 2 ** np.arange(6):
            img_resized = Preprocess.rescale(img.copy(), (WIDTH // i, HEIGHT // i))
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

    def get_dataset_by_scale(self, img, label, plot, i):
        img_resized = Preprocess.rescale(img, (WIDTH // i, HEIGHT // i))
        img_lbp = self.apply_lbp(img_resized, plot=plot)
        img_lbp = Preprocess.repeat_pixels(img_lbp, i)
        if label is not None:
            label_resized = Preprocess.rescale(label.copy(), (WIDTH // i, HEIGHT // i))
            label_resized = Preprocess.repeat_pixels(label_resized, i)
            if plot:
                self.plot_preprocess_with_label(img_lbp.reshape(-1, 1), label_resized.reshape(-1, 1), None)
            return pd.DataFrame(np.concatenate((img_lbp.reshape(-1, 1), label_resized.reshape(-1, 1)), axis=1))
        else:
            return pd.DataFrame(img_lbp.reshape(-1, 1))

    def get_datasets_by_scale(self, path, label_path=None, mask_path=None, balance=False, plot=False):
        img = Preprocess.read_img(path)
        img, mask = self.filter_by_mask(img, mask_path)
        img = self.rescale_add_borders(img)
        if label_path is not None:
            label = self.get_label(label_path)
            label = self.rescale_add_borders(label)
        else:
            label = None
        dfs = [self.get_dataset_by_scale(img.copy(), label, plot, i) for i in 2 ** np.arange(6)]
        _, selected_indexes = self.remove_mask_data(np.array(dfs[0]), mask, remove_borders=True)
        if balance:
            aux = np.random.choice(np.where(dfs[0].iloc[selected_indexes, -1] == 0)[0],
                                   size=sum(dfs[0].iloc[selected_indexes, -1] == 1),
                                   replace=False)
            aux = np.concatenate((aux, np.where(dfs[0].iloc[selected_indexes, -1] == 1)[0]))
            selected_indexes = selected_indexes[np.sort(aux)]
        return [df.iloc[selected_indexes, :] for df in dfs]

    def get_label(self, path):
        img = Preprocess.read_img(path)
        img[img < LABEL_THRESHOLD] = 0
        img[img > LABEL_THRESHOLD] = 1
        return img

    def plot_preprocess_with_label(self, img, label, mask):
        def array_to_mat(arr, mask_mat):
            mat = np.copy(mask_mat)
            mat[mat < MASK_THRESHOLD] = 0
            mat[mat >= MASK_THRESHOLD] = arr.ravel()
            return mat

        if mask is None:
            mask = np.ones((HEIGHT, WIDTH)) * MASK_THRESHOLD
        img = array_to_mat(img, mask)
        img = (img * (255 / np.max(img))).astype(int)
        label = array_to_mat(label, mask)
        img = np.asarray(PIL.Image.fromarray(np.uint8(img)).convert('RGB')).copy()
        img[label == 1] = [255, 0, 0]
        im = PIL.Image.fromarray(np.uint8(img))
        im.show()
