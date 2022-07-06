PATH = r'/home/fer/Drive/Estudios/Master-IA/TFM/dataset/STARE/training/images'
path = PATH + '/im0319.ppm'
#
from os import listdir
import cv2
import numpy as np
import pandas as pd
from PIL import Image
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import f1_score
from preprocess.preprocess import Preprocess
#
img = Preprocess.img_processing(np.asarray(Image.open(path).convert('RGB'))[:, :, 0])
# img_k1 = cv2.filter2D(img, -1, k1)
# img_k2 = cv2.filter2D(img, -1, k2)
import matplotlib.pyplot as plt
# from PIL import Image
# from fitness import IMAGES
#
img0 = Preprocess.img_processing(np.asarray(Image.open(path).convert('RGB'))[:, :, 0])
img1 = Preprocess.img_processing(np.asarray(Image.open(path).convert('RGB'))[:, :, 1])
img2 = Preprocess.img_processing(np.asarray(Image.open(path).convert('RGB'))[:, :, 2])
im0 = Image.fromarray(np.uint8(img0))
plt.figure(figsize=(15, 11), dpi=80)
plt.imshow(im0, cmap='gray')
plt.show()
im1 = Image.fromarray(np.uint8(img1))
plt.figure(figsize=(15, 11), dpi=80)
plt.imshow(im1, cmap='gray')
plt.show()
im2 = Image.fromarray(np.uint8(img2))
plt.figure(figsize=(15, 11), dpi=80)
plt.imshow(im2, cmap='gray')
plt.show()
