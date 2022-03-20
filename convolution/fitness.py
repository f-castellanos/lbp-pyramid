from os import listdir

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score

from preprocess.preprocess import Preprocess

CLF = MultinomialNB(fit_prior=True)
PATH = r'/home/fer/Drive/Estudios/Master-IA/TFM/dataset/DRIVE/training/images'
MASK_PATH = r'/home/fer/Drive/Estudios/Master-IA/TFM/dataset/DRIVE/training/mask'
LABELS_PATH = r'/home/fer/Drive/Estudios/Master-IA/TFM/dataset/DRIVE/training/1st_manual'


def load_images():
    paths = [f"{PATH}/{path}" for path in sorted(listdir(PATH))][:14]
    return [Preprocess.img_processing(np.asarray(Image.open(path).convert('RGB'))[:, :, 1])
            for path in paths]


def load_masks():
    paths = [f"{MASK_PATH}/{path}" for path in sorted(listdir(MASK_PATH))][:14]
    return [np.asarray(Image.open(path).convert('L')) > 100 for path in paths]


def load_labels():
    paths = [f"{LABELS_PATH}/{path}" for path in sorted(listdir(LABELS_PATH))][:14]
    return [np.asarray(Image.open(path).convert('L')) > 30 for path in paths]


IMAGES = load_images()
MASKS = load_masks()
Y_TRAIN = pd.DataFrame(
    np.concatenate([np.array(label, dtype=int)[MASKS[i]].ravel() for i, label in enumerate(load_labels()) if i < 10],
                   axis=0).T).values.ravel()
Y_TEST = pd.DataFrame(
    np.concatenate([np.array(label, dtype=int)[MASKS[i]].ravel() for i, label in enumerate(load_labels()) if i >= 10],
                   axis=0).T).values.ravel()


def fitness_function(individual, n_kernels):
    k_len = int(len(individual)/n_kernels)
    k_size = int(np.sqrt(k_len))
    features = [pd.DataFrame(np.array(
        [cv2.filter2D(img, -1, individual[i*k_len:(i+1)*k_len].reshape((k_size, k_size)))[mask]
         for i in range(n_kernels)]
    ).T)
        for img, mask in zip(IMAGES, MASKS)]
    CLF.fit(pd.concat(features[:10], ignore_index=True), Y_TRAIN)
    y_pred = CLF.predict(pd.concat(features[10:], ignore_index=True))
    return f1_score(Y_TEST, y_pred)

# def fitness_function(x, *_, **__):
#     """
#     Six-Hump Camel-Back Function
#     https://towardsdatascience.com/unit-3-genetic-algorithm-benchmark-test-function-1-670a55088064
#     """
#     x1 = x[0]
#     x2 = x[1]
#     return (4 - 2.1 * np.power(x1, 2) + np.power(x1, 4) / 3) * np.power(x1, 2) + x1 * x2 + (-4 + 4 * np.power(x2, 2)) * np.power(x2, 2)
