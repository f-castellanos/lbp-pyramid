# https://www.kaggle.com/datasets/srinjoybhuiya/drive-retinal-vessel-segmentation-pixelwise

import pandas as pd
import os
import numpy as np
from preprocess.preprocess import Preprocess
from PIL import Image

csv_folder = '/home/fer/Drive/Estudios/Master-IA/TFM/dataset/DRIVE_Modified/Modified_dataset_test'
csv_paths = sorted(os.listdir(csv_folder))

img_folder = '/home/fer/Drive/Estudios/Master-IA/TFM/dataset/DRIVE/test/images'
img_paths = sorted(os.listdir(img_folder))
mask_folder = '/home/fer/Drive/Estudios/Master-IA/TFM/dataset/DRIVE/test/mask'
masks_paths = sorted(os.listdir(mask_folder))

preprocess = Preprocess(
    height={'DRIVE': 608, 'CHASE': 960, 'STARE': 608}['DRIVE'],
    width={'DRIVE': 576, 'CHASE': 1024, 'STARE': 704}['DRIVE']
)
sa = 20 if 'DRIVE' == 'CHASE' else 14
sb = 28 if 'DRIVE' == 'CHASE' else 20

for csv_path, img_path, mask_path in zip(csv_paths, img_paths, masks_paths):
    y = pd.read_csv(f"{csv_folder}/{csv_path}")['truth labels']

    # label_predicted = (np.array(y) == 255).astype(int)
    # # label_predicted = np.array(y).reshape(584, 565)
    #
    # img = preprocess.read_img(f"{img_folder}/{img_path}").ravel()
    # mask = preprocess.read_img(f"{mask_folder}/{mask_path}")
    # img = img[mask.ravel() > 100]
    # label_predicted = label_predicted[mask.ravel() > 100]
    # preprocess.plot_preprocess_with_label(img, label_predicted, mask)

    y = np.array(y).reshape(584, 565)
    im = Image.fromarray(np.uint8(y))
    im.save(f"/home/fer/Drive/Estudios/Master-IA/TFM/dataset/DRIVE/test/1st_manual/{img_path}".replace('.tif', '.png'))
    a = 0
