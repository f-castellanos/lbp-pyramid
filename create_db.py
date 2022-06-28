##
import os
import pickle
# import zipfile
from pathlib import Path
import itertools

import pandas as pd
from joblib import Parallel, delayed, parallel_backend

import PARAMETERS
from plot_results.plot_results import PlotResults
from preprocess.preprocess import Preprocess, VALID_PARAMETERS


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
    if i < {'DRIVE': 14, 'CHASE': 20, 'STARE': 14}[PARAMETERS.DATASET]:
        train_set = True
    else:
        train_set = False
    df_img = getattr(preprocess, PARAMETERS.METHOD)(
        img_path, label_path=label_path, mask_path=mask_path, train_set=train_set, plot=PARAMETERS.PLOT
    )
    return df_img


def main(single_exec=False):
    # Object initialization
    preprocess = Preprocess(
        lbp_radius=1,
        lbp_method=PARAMETERS.LBP_METHOD,
        # height=PARAMETERS.HEIGHT,
        height={'DRIVE': 608, 'CHASE': 960, 'STARE': 608}[PARAMETERS.DATASET],
        width={'DRIVE': 576, 'CHASE': 1024, 'STARE': 704}[PARAMETERS.DATASET],
        # width=PARAMETERS.WIDTH,
        balance=PARAMETERS.BALANCE
    )
    _ = PlotResults(height=preprocess.height, width=preprocess.width)
    # DB folders
    parent_path = str(Path(os.path.dirname(os.path.abspath(__file__))).parent)
    path = parent_path + f'/dataset/{PARAMETERS.DATASET}/training/'
    images_path = path + 'images/'
    images = sorted(os.listdir(images_path))
    masks_path = path + 'mask/'
    masks = sorted(os.listdir(masks_path))
    labels_path = path + '1st_manual/'
    labels = sorted(os.listdir(labels_path))
    preprocess.compute_preprocessing(images, masks, path)

    # Train - Test dataframes
    if PARAMETERS.CONVOLUTION is None and PARAMETERS.RADIUS == 1 and PARAMETERS.CHANNEL is None and not PARAMETERS.PREPROCESS_OPTIMIZATION:
        db_folder = f'DB/{PARAMETERS.DATASET}'
    elif PARAMETERS.PREPROCESS_OPTIMIZATION:
        channels_map = {0: 'red', 1: 'green', 2: 'blue'}
        db_folder = \
            f'DB/{PARAMETERS.DATASET}/extra_features/preprocess_optimization_{channels_map[PARAMETERS.CHANNEL]}' + {
                "default": "", "gb": "_gb", "w": "_w", "lbp": "_lbp", 'lbp_gb': "_lbp_gb"}[PARAMETERS.PREPROCESS_TYPE]
    elif PARAMETERS.CONVOLUTION is None and PARAMETERS.RADIUS > 1:
        db_folder = f'DB/{PARAMETERS.DATASET}/extra_features/radius/{PARAMETERS.RADIUS}'
    elif PARAMETERS.CHANNEL is not None and PARAMETERS.CONVOLUTION is None:
        channels_map = {0: 'red', 1: 'green', 2: 'blue'}
        db_folder = f"DB/{PARAMETERS.DATASET}/extra_features/rgb/{channels_map[PARAMETERS.CHANNEL]}"
    else:
        db_folder = f'DB/{PARAMETERS.DATASET}/extra_features/convolution/{PARAMETERS.CONV_PATH}'
    db_path = f"{parent_path}/{db_folder}"
    if not os.path.exists(db_path):
        os.makedirs(db_path)
    train_file_name = f"{db_path}/train_train_{PARAMETERS.FILE_EXTENSION}"
    test_file_name = f"{db_path}/train_test_{PARAMETERS.FILE_EXTENSION}"

    if single_exec:
        _ = img_preprocess(preprocess, 0, images[0], masks[0], labels[0], path)
    elif (not (os.path.isfile(f'{train_file_name}.pkl') and os.path.isfile(f'{test_file_name}.pkl'))) or (
            'FORCE_EXEC' in os.environ and os.environ['FORCE_EXEC'] == 'True'):

        # Image iteration
        with parallel_backend('multiprocessing', n_jobs=PARAMETERS.N_JOBS):
            df_list = Parallel()(
                delayed(img_preprocess)(preprocess, i, image, mask, label, path)
                for i, (image, mask, label) in enumerate(zip(images, masks, labels))
            )

        if PARAMETERS.METHOD == 'get_datasets_by_scale':
            def train_set_extract(df_set, i):
                df = pd.DataFrame()
                for dfs in df_set:
                    df = pd.concat((df, dfs['datasets'][i]), axis=0)
                return df
            with open(f'{train_file_name}.pkl', 'wb') as f:
                pickle.dump([train_set_extract(df_list[:14], i) for i in range(len(df_list[0]['datasets']))], f)
            with open(f'{test_file_name}.pkl', 'wb') as f:
                pickle.dump(df_list[14:], f)
        elif PARAMETERS.METHOD == 'get_pyramid_dataset':
            pd.concat(df_list[:14]).to_pickle(f'{train_file_name}.pkl', compression='gzip')
            pd.concat(df_list[14:]).to_pickle(f'{test_file_name}.pkl', compression='gzip')


##
if __name__ == '__main__':
    if 'GRID_SEARCH' not in os.environ or os.environ['GRID_SEARCH'] != 'TRUE':
        main()
    else:
        j = 0
        dict_keys = list(VALID_PARAMETERS.keys())
        for combination in itertools.product(*[VALID_PARAMETERS[k] for k in dict_keys]):
            print(combination)
            j += 1
            print(j)
            for i, v in enumerate(combination):
                setattr(PARAMETERS, dict_keys[i], v)
                PARAMETERS.FILE_EXTENSION = f"{PARAMETERS.LBP_METHOD}_{PARAMETERS.METHOD}_" \
                                            f"{PARAMETERS.INTERPOLATION_ALGORITHM}_balance-{PARAMETERS.BALANCE}_" \
                                            f"scales-{PARAMETERS.N_SCALES}_x2-{PARAMETERS.X2SCALE}" \
                                            f"_gray-intensity-{PARAMETERS.GRAY_INTENSITY}"
            if j > 0 and not (PARAMETERS.N_SCALES == 1 and PARAMETERS.X2SCALE):
                main()
