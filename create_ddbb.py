##
from preprocess.preprocess import Preprocess
from plot_results.plot_results import PlotResults
import os
from pathlib import Path
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
import pickle
import zipfile


PLOT = True
BALANCE = False
# LBP_METHOD = 'default'
# LBP_METHOD = 'riu'
LBP_METHOD = 'riu2'
# METHOD = 'get_pyramid_dataset'
METHOD = 'get_datasets_by_scale'
# N_JOBS = 1
N_JOBS = 1


def img_preprocess(preprocess,
                   i, image, mask, label, path,
                   images_path='images/',
                   masks_path='mask/',
                   labels_path='1st_manual/'):
    # Image path
    img_path = path + images_path + image
    mask_path = path + masks_path + mask
    label_path = path + labels_path + label
    # Image processing
    if i < 14:
        train_set = True
    else:
        train_set = False
    df_img = getattr(preprocess, METHOD)(
        img_path, label_path=label_path, mask_path=mask_path, train_set=train_set, plot=PLOT
    )
    return df_img


def main():
    # Object initialization
    preprocess = Preprocess(lbp_radius=1, lbp_method=LBP_METHOD, height=608, width=576, balance=BALANCE)
    _ = PlotResults(height=608, width=576)
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
    with parallel_backend('multiprocessing', n_jobs=N_JOBS):
        df_list = Parallel()(
            delayed(img_preprocess)(preprocess, i, image, mask, label, path)
            for i, (image, mask, label) in enumerate(zip(images, masks, labels))
        )

    # Train - Test dataframes
    train_file_name = parent_path + '/DB/train_train_' + LBP_METHOD + '_' + METHOD
    test_file_name = parent_path + '/DB/train_test_' + LBP_METHOD + '_' + METHOD
    if METHOD == 'get_datasets_by_scale':
        def train_set_extract(df_set, i):
            df = pd.DataFrame()
            for dfs in df_set:
                df = pd.concat((df, dfs['datasets'][i]), axis=0)
            return df
        with open(f'{train_file_name}.pkl', 'wb') as f:
            pickle.dump([train_set_extract(df_list[:14], i) for i in range(len(df_list[0]['datasets']))], f)
        with open(f'{test_file_name}.pkl', 'wb') as f:
            pickle.dump(df_list[14:], f)
    elif METHOD == 'get_pyramid_dataset':
        pd.concat(df_list[:14]).to_pickle(f'{train_file_name}.pkl')
        pd.concat(df_list[14:]).to_pickle(f'{test_file_name}.pkl')

    # Zip output
    try:
        os.remove(f'{train_file_name}.zip')
    except OSError:
        pass
    zipfile.ZipFile(f'{train_file_name}.zip', 'w', zipfile.ZIP_DEFLATED).write(
        f'{train_file_name}.pkl',
        f'{train_file_name.split("/")[-1]}.pkl'
    )
    os.remove(f'{train_file_name}.pkl')
    try:
        os.remove(f'{test_file_name}.zip')
    except OSError:
        pass
    zipfile.ZipFile(f'{test_file_name}.zip', 'w', zipfile.ZIP_DEFLATED).write(
        f'{test_file_name}.pkl',
        f'{test_file_name.split("/")[-1]}.pkl'
    )
    os.remove(f'{test_file_name}.pkl')


##
if __name__ == '__main__':
    main()

else:
    ##
    N_JOBS = -1
    for LBP_METHOD in ['riu2', 'riu']:
        for METHOD in ['get_pyramid_dataset', 'get_datasets_by_scale']:
            main()
