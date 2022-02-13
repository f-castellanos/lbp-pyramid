import os
from PIL import Image
import numpy as np
import cv2


CHASE_PATH = '/home/fer/Drive/Estudios/Master-IA/TFM/dataset/CHASE/training'

image_paths = sorted(os.listdir(f"{CHASE_PATH}/images"))

for filename in image_paths:
    img = np.asarray(Image.open(f"{CHASE_PATH}/images/{filename}").convert('L'))
    img[img > 5] = 255
    img[img < 255] = 0
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((10, 10)))
    im = Image.fromarray(img)
    im.save(f"{CHASE_PATH}/mask/{filename}")
