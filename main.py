from preprocess.preprocess import Preprocess
from plot_results.plot_results import PlotResults
import os
from pathlib import Path
import pandas as pd


parent_path = str(Path(os.path.dirname(os.path.abspath(__file__))).parent)
preprocess = Preprocess(height=608, width=576)
plot_results = PlotResults(height=608, width=576)
images_path = parent_path + '/dataset/training/images/'
images = sorted(os.listdir(images_path))
masks_path = parent_path + '/dataset/training/mask/'
masks = sorted(os.listdir(masks_path))
labels_path = parent_path + '/dataset/training/1st_manual/'
df = pd.DataFrame()
labels = sorted(os.listdir(labels_path))
for image, mask, label in zip(images, masks, labels):
    img_path = images_path + image
    mask_path = masks_path + mask
    label_path = labels_path + label
    df_img = preprocess.get_features(img_path, mask_path=mask_path, plot=False)
    df_img['label'] = preprocess.get_label(label_path)
    df = pd.concat((df, df_img))
# df.to_pickle(parent_path + '/DDBB/train.pkl')
