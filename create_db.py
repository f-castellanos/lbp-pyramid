##
import os
import pickle
import zipfile
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
    if i < 14:
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
        height=PARAMETERS.HEIGHT,
        width=PARAMETERS.WIDTH,
        balance=PARAMETERS.BALANCE
    )
    _ = PlotResults(height=preprocess.height, width=preprocess.width)
    # DB folders
    parent_path = str(Path(os.path.dirname(os.path.abspath(__file__))).parent)
    path = parent_path + '/dataset/training/'
    images_path = path + 'images/'
    images = sorted(os.listdir(images_path))
    masks_path = path + 'mask/'
    masks = sorted(os.listdir(masks_path))
    labels_path = path + '1st_manual/'
    labels = sorted(os.listdir(labels_path))

    if single_exec:
        _ = img_preprocess(preprocess, 0, images[0], masks[0], labels[0], path)
    else:

        # Image iteration
        with parallel_backend('multiprocessing', n_jobs=PARAMETERS.N_JOBS):
            df_list = Parallel()(
                delayed(img_preprocess)(preprocess, i, image, mask, label, path)
                for i, (image, mask, label) in enumerate(zip(images, masks, labels))
            )

        # Train - Test dataframes
        train_file_name = f"{parent_path}/DB/train_train_{PARAMETERS.FILE_EXTENSION}"
        test_file_name = f"{parent_path}/DB/train_test_{PARAMETERS.FILE_EXTENSION}"
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
            if j > 0:
                main()

# Saltados: 98, 100, 101, 102, 104, 105, 106, 107
