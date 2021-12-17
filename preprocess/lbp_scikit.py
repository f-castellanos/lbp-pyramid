from skimage.feature import local_binary_pattern
from pathlib import Path
import os
import matplotlib.pyplot as plt
import itertools
import numpy as np

PARENT_PATH = str(Path(os.path.dirname(os.path.abspath(__file__))).parent.parent)
LBP_METHODS_MAP = {
    'default': 'default',
    'riu': 'ror',
    'riu2': 'uniform',
    'nriuniform': 'nri_uniform',
    'var': 'var',
}


def get_value_map(n_points):
    possible_values = []
    for i in range(n_points + 1):
        permutations = set(itertools.permutations([1] * i + [0] * (n_points - i)))
        for permutation in permutations:
            rotations = np.arange(len(permutation))
            arr_mat = np.array(permutation).reshape(1, -1)
            rows, column_indices = np.ogrid[:arr_mat.shape[0], :arr_mat.shape[1]]
            rotations[rotations < 0] += arr_mat.shape[1]
            column_indices = column_indices - rotations[:, np.newaxis]
            rolled_arr = arr_mat[rows, column_indices]
            decimal_conversion = np.repeat((2 ** np.arange(len(permutation))[::-1]).reshape(1, -1), len(permutation), axis=0)  # noqa
            possible_values.append((decimal_conversion * rolled_arr).sum(axis=1).min())
    possible_values = sorted(set(possible_values))
    return {k: v for v, k in enumerate(possible_values)}


VALUES_MAP = {
    'riu': {
        8: get_value_map(8)
    }
}


def roll_pattern(pattern, k):
    rolled = np.roll(pattern, k)
    return sum(rolled * (2 ** np.arange(len(rolled))[::-1]))


def plot_lbp(img, img_lbp):
    plt.figure(figsize=(16, 12))
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(img_lbp, cmap='gray', vmax=np.max(img_lbp))
    plt.title('LBP Image')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    bins = np.arange(np.max(img_lbp)+2)-0.5
    plt.hist(img_lbp.ravel(), bins=np.arange(np.max(img_lbp)+2)-0.5, ec='white')
    plt.xlim(bins[0], bins[-1])
    plt.title('Histogram')
    plt.show()


def lbp(img, r=1, method='default', plot=False):
    # p is the number of pixels in a square of radio r
    p = min(8 * r, 32)
    n_points = p * r
    img_lbp = local_binary_pattern(img, p, r, method=LBP_METHODS_MAP[method])
    if method == 'riu':
        if n_points in VALUES_MAP['riu']:
            d = VALUES_MAP['riu'][n_points]
        else:
            d = get_value_map(n_points)
        img_lbp = np.vectorize(d.get)(img_lbp)
    if plot:
        plot_lbp(img, img_lbp)
    return img_lbp


# import sys
# sys.path.append(os.path.dirname(f"{PARENT_PATH}/lbp-pyramid/preprocess"))
# from preprocess import Preprocess
# img = Preprocess.read_img(f"{PARENT_PATH}/dataset/training/preprocessed/lanczos/original/21_training_1.0.jpeg")
# img = lbp(img, method='riu2')
# a = 0
