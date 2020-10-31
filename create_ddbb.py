from preprocess.preprocess import Preprocess
from plot_results.plot_results import PlotResults
import os
from pathlib import Path
import pandas as pd


# Object initialization
preprocess = Preprocess(height=608, width=576)
plot_results = PlotResults(height=608, width=576)
# DB folders
parent_path = str(Path(os.path.dirname(os.path.abspath(__file__))).parent)
images_path = parent_path + '/dataset/training/images/'
images = sorted(os.listdir(images_path))
masks_path = parent_path + '/dataset/training/mask/'
masks = sorted(os.listdir(masks_path))
labels_path = parent_path + '/dataset/training/1st_manual/'
labels = sorted(os.listdir(labels_path))
# DataFrame initialization
df = pd.DataFrame()
# Image iteration
for i, (image, mask, label) in enumerate(zip(images, masks, labels)):
    # Image path
    img_path = images_path + image
    mask_path = masks_path + mask
    label_path = labels_path + label
    # Image processing
    df_img = preprocess.get_features(img_path, mask_path=mask_path, plot=True)
    df_img['label'] = preprocess.get_label(label_path)
    # Image concatenation
    df = pd.concat((df, df_img))
    if i == 13:
        # Train dataset
        df.to_pickle(parent_path + '/DDBB/train_train.pkl')
        df = pd.DataFrame()
    elif i == 19:
        # Test dataset
        df.to_pickle(parent_path + '/DDBB/train_test.pkl')
