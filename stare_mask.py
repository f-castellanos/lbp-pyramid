import os
from PIL import Image
import numpy as np
import cv2


STARE_PATH = '/home/fer/Drive/Estudios/Master-IA/TFM/dataset/STARE/training'

image_paths = sorted(os.listdir(f"{STARE_PATH}/images"))
kernel = np.ones((5,5),np.float32)/25
# kernel = np.ones((5,5),np.float32)/25
for filename in image_paths:
    img = np.asarray(Image.open(f"{STARE_PATH}/images/{filename}").convert('L'))

    img = cv2.filter2D(img, -1, kernel)
    cla_he = cv2.createCLAHE(clipLimit=.02, tileGridSize=(8, 8))
    img = cla_he.apply(img)
    img = cv2.convertScaleAbs(img, alpha=3, beta=100)


    # img = img[:, tuple(range(500, 700))]
    # img = cv2.convertScaleAbs(img, alpha=1, beta=90)

    # cv2.imshow('original', img)
    # img[img > 5] = 255
    # cla_he = cv2.createCLAHE(clipLimit=20, tileGridSize=(3, 3))
    print(np.percentile(img, 25), np.percentile(img, 26))
    img[img < min(np.percentile(img, 26) + 20,250)] = 0
    img[img > 0] = 255
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((10, 10)))
    im = Image.fromarray(img)
    im.save(f"{STARE_PATH}/mask/{filename}")
