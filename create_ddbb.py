##
from preprocess.preprocess import Preprocess
from plot_results.plot_results import PlotResults
import os
from pathlib import Path
import pandas as pd
from joblib import Parallel, delayed, parallel_backend


def img_preprocess(i, image, mask, label, path,
                   images_path='images/',
                   masks_path='mask/',
                   labels_path='1st_manual/'):
    # Image path
    img_path = path + images_path + image
    mask_path = path + masks_path + mask
    label_path = path + labels_path + label
    # Image processing
    if i < 14:
        balance = True
    else:
        balance = False
    df_img = preprocess.get_dataset(img_path, label_path=label_path, mask_path=mask_path, balance=balance, plot=False)
    return df_img


if __name__ == '__main__':
    # Object initialization
    lbp_method = 'riu2'
    preprocess = Preprocess(lbp_radius=1, lbp_method=lbp_method, height=608, width=576)
    plot_results = PlotResults(height=608, width=576)
    # DB folders
    parent_path = str(Path(os.path.dirname(os.path.abspath(__file__))).parent)
    path = parent_path + '/dataset/training/'
    images_path = path + 'images/'
    images = sorted(os.listdir(images_path))
    masks_path = path + 'mask/'
    masks = sorted(os.listdir(masks_path))
    labels_path = path + '1st_manual/'
    labels = sorted(os.listdir(labels_path))

    # Image iteration
    with parallel_backend('multiprocessing', n_jobs=-1):
        df_list = Parallel()(
            delayed(img_preprocess)(i, image, mask, label, path)
            for i, (image, mask, label) in enumerate(zip(images, masks, labels))
        )

    # Train - Test dataframes
    pd.concat(df_list[:14]).to_pickle(parent_path + '/DB/train_train_' + lbp_method + '.pkl')
    pd.concat(df_list[14:]).to_pickle(parent_path + '/DB/train_test_' + lbp_method + '.pkl')
