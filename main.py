##
import os
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import CategoricalNB, MultinomialNB
# https://ichi.pro/es/por-que-y-como-utilizar-los-algoritmos-naive-bayes-en-una-industria-regulada-con-sklearn-python-codigo-241721569620687  noqa
# from xgboost import XGBClassifier
from sklearn.metrics import f1_score, make_scorer

import lightgbm

from confusion_matrix_pretty_print import print_confusion_matrix
from preprocess.preprocess import Preprocess
import PARAMETERS


class LGBMCategorical(lightgbm.LGBMClassifier):
    def fit(self, x, *args, **kwargs):
        x.columns = [str(col).replace(':', '') for col in x.columns]
        for col in x.columns:
            x[col] = x[col].astype('category')
        return super().fit(x, *args, **kwargs)

    def predict(self, x, *args, **kwargs):
        x.columns = [str(col).replace(':', '') for col in x.columns]
        for col in x.columns:
            x[col] = x[col].astype('category')
        return super().predict(x, *args, **kwargs)


class LGBMNumerical(lightgbm.LGBMClassifier):
    def fit(self, x, *args, **kwargs):
        x.columns = [str(col).replace(':', '') for col in x.columns]
        for col in x.columns:
            x[col] = x[col].astype('float')
        return super().fit(x, *args, **kwargs)

    def predict(self, x, *args, **kwargs):
        x.columns = [str(col).replace(':', '') for col in x.columns]
        for col in x.columns:
            x[col] = x[col].astype('float')
        return super().predict(x, *args, **kwargs)


class LGBMCatNum(lightgbm.LGBMClassifier):
    def fit(self, x, *args, **kwargs):
        x.columns = [str(col).replace(':', '') for col in x.columns]
        for col in x.columns:
            if len(np.unique(x[col])) < 260:
                x[col] = x[col].astype('category')
            else:
                x[col] = x[col].astype('float')
        return super().fit(x, *args, **kwargs)

    def predict(self, x, *args, **kwargs):
        x.columns = [str(col).replace(':', '') for col in x.columns]
        for col in x.columns:
            if len(np.unique(x[col])) < 260:
                x[col] = x[col].astype('category')
            else:
                x[col] = x[col].astype('float')
        return super().predict(x, *args, **kwargs)


def init_clf_and_fit(df, y, lgb=''):
    if lgb == 'Num':
        # clf = XGBClassifier(n_jobs=-1, objective='binary:logistic', enable_categorical=True)
        # f1 = make_scorer(f1_score, average='macro')
        # parameters = {
        #     'num_leaves': [10, 15, 30],
        #     'min_child_samples': [5, 10, 15],
        #     'max_depth': [5, 10, 20],
        #     'learning_rate': [0.05, 0.1],
        #     # 'reg_alpha': [0, 0.01, 0.03],
        #     'colsample_bytree': [0.4, 0.6, 0.8],
        #     'subsample': [0.6, 0.8]
        # }
        clf = LGBMNumerical(
            num_leaves=50,
            max_depth=30,
            random_state=42,
            verbose=0,
            # metric='None',
            n_jobs=8,
            # n_estimators=1000,
            colsample_bytree=0.9,
            subsample=0.7,
            learning_rate=0.5
        )
        # clf = GridSearchCV(lgb_clf, parameters, n_jobs=-1, scoring=f1)
        clf.fit(df, y, eval_metric=['auc'])
    elif lgb == 'Cat':
        clf = LGBMCategorical(
            num_leaves=50,
            max_depth=30,
            random_state=42,
            verbose=0,
            # metric='None',
            n_jobs=8,
            # n_estimators=1000,
            colsample_bytree=0.9,
            subsample=0.7,
            learning_rate=0.5
        )
        clf.fit(df, y, eval_metric=['auc'])
    elif lgb == 'CatNum':
        clf = LGBMCatNum(
            num_leaves=50,
            max_depth=30,
            random_state=42,
            verbose=0,
            # metric='None',
            n_jobs=8,
            # n_estimators=1000,
            colsample_bytree=0.9,
            subsample=0.7,
            learning_rate=0.5
        )
        clf.fit(df, y, eval_metric=['auc'])
    else:
        # if 'Original' in df:
        #     multi_clf = MultinomialNB(fit_prior=True)
        #     multi_clf.fit(df[['Original']], y)
        #     df.drop(columns=['Original'], inplace=True)
        if PARAMETERS.LBP_METHOD == 'var':
            clf = MultinomialNB(fit_prior=True)
        else:
            clf = CategoricalNB(
                fit_prior=True,
                min_categories={
                    'default': 256,
                    'riu': 36,
                    'riu2': 10,
                    'nriuniform': 59,
                }[PARAMETERS.LBP_METHOD]
            )
        # if 'Original' in df:
        #     clf = RandomForestClassifier(estimators=[('multi', multi_clf), ('cat', clf)])
        clf.fit(df, y)
        with open(f"lbm_fit.pkl", 'wb') as f:
            pickle.dump(clf, f)
    return clf


def ensemble_prediction(classifiers, dfs_test):
    # Calculating predictions
    y_predicted_list = [classifier.predict(test_set.iloc[:, :-1]).reshape(-1, 1)
                        for classifier, test_set in zip(classifiers, dfs_test['datasets'])]
    # Equaling the number of pixels
    y_predicted_list = [y_predicted_list[i].reshape(PARAMETERS.HEIGHT//(2 ** i), -1)
                        for i in range(len(y_predicted_list))]
    y_predicted_list = [Preprocess.repeat_pixels(prediction, (2 ** i))
                        for i, prediction in enumerate(y_predicted_list)]
    # Mask application
    y_predicted_list = [prediction.ravel()[dfs_test['mask']].reshape(-1, 1) for prediction in y_predicted_list]
    # print(np.mean(np.concatenate(y_predicted_list, axis=1), axis=0))
    # Predictions
    y_predictions = np.mean(np.concatenate(y_predicted_list, axis=1), axis=1)
    # Actual values
    y_actual = np.array(dfs_test['datasets'][0].iloc[:, -1]).ravel()[dfs_test['mask']]
    return y_predictions, y_actual


def load_datasets_for_lbp_operator(parent_path, discard_columns=False):
    # Database reading
    db_folder = 'DB'
    # if PARAMETERS.CONVOLUTION is None and PARAMETERS.RADIUS == 1:
    #     db_folder = 'DB'
    # elif PARAMETERS.CONVOLUTION is None and PARAMETERS.RADIUS > 1:
    #     db_folder = f'DB/extra_features/radius/{PARAMETERS.RADIUS}'
    # else:
    #     db_folder = f'DB/extra_features/convolution/{PARAMETERS.CONVOLUTION}'
    db_path = f"{parent_path}/{db_folder}"
    train_file_name = f"{db_path}/train_train_{PARAMETERS.FILE_EXTENSION}"
    test_file_name = f"{db_path}/train_test_{PARAMETERS.FILE_EXTENSION}"
    df_train_temp = pd.read_pickle(f'{train_file_name}.pkl', compression='gzip')
    df_test_temp = pd.read_pickle(f'{test_file_name}.pkl', compression='gzip')

    y_train_temp = df_train_temp.loc[:, 'label']
    y_test_temp = df_test_temp.loc[:, 'label']
    df_train_temp.drop(columns=['label'], inplace=True)
    df_test_temp.drop(columns=['label'], inplace=True)

    if discard_columns and PARAMETERS.GRAY_INTENSITY:
        df_train_temp.drop(columns=['Original'], inplace=True)
        df_test_temp.drop(columns=['Original'], inplace=True)
    df_train_temp.columns = list(
        map(lambda x: f"{PARAMETERS.LBP_METHOD}_{x}" if x != 'Original' else x, df_train_temp.columns))
    df_test_temp.columns = list(
        map(lambda x: f"{PARAMETERS.LBP_METHOD}_{x}" if x != 'Original' else x, df_test_temp.columns))

    for col in df_train_temp.columns:
        df_train_temp[col] = df_train_temp[col].astype('category')
        df_test_temp[col] = df_test_temp[col].astype('category')

    return df_train_temp, df_test_temp, y_train_temp, y_test_temp


def main(lgb='', plot_once=False, extra_features=None, all_lbp=False):
    parent_path = str(Path(os.path.dirname(os.path.abspath(__file__))).parent)
    flag = True
    if PARAMETERS.METHOD == 'get_datasets_by_scale':
        # Database reading
        db_folder = 'DB'
        # if PARAMETERS.CONVOLUTION is None and PARAMETERS.RADIUS == 1:
        #     db_folder = 'DB'
        # elif PARAMETERS.CONVOLUTION is None and PARAMETERS.RADIUS > 1:
        #     db_folder = f'DB/extra_features/radius/{PARAMETERS.RADIUS}'
        # else:
        #     db_folder = f'DB/extra_features/convolution/{PARAMETERS.CONVOLUTION}'
        db_path = f"{parent_path}/{db_folder}"
        train_file_name = f"{db_path}/train_train_{PARAMETERS.FILE_EXTENSION}"
        test_file_name = f"{db_path}/train_test_{PARAMETERS.FILE_EXTENSION}"
        with zipfile.ZipFile(f'{train_file_name}.zip', 'r') as zip_ref:
            zip_ref.extractall(f'{db_path}')

        with zipfile.ZipFile(f'{test_file_name}.zip', 'r') as zip_ref:
            zip_ref.extractall(f'{db_path}')

        with open(f'{train_file_name}.pkl', 'rb') as f:
            df_train_list = pickle.load(f)
        os.remove(f'{train_file_name}.pkl')
        clf_list = [init_clf_and_fit(df_train.iloc[:, :-1], df_train.iloc[:, -1], lgb) for df_train in df_train_list]
        with open(f'{test_file_name}.pkl', 'rb') as f:
            df_test_list = pickle.load(f)
        os.remove(f'{test_file_name}.pkl')

        y_predicted_test_list = [ensemble_prediction(clf_list, dfs_test) for dfs_test in df_test_list]
        y_predicted = np.concatenate([y_predicted_test_pic[0] for y_predicted_test_pic in y_predicted_test_list])
        y_predicted = np.where(y_predicted > 0.5, 1, 0)
        y_test = np.concatenate([y_predicted_test_pic[1] for y_predicted_test_pic in y_predicted_test_list])

    else:
        df_train = None
        df_test = None
        if all_lbp:
            for i, lbp_operator in enumerate(['default', 'riu', 'riu2', 'nriuniform', 'var']):
                PARAMETERS.LBP_METHOD = lbp_operator
                PARAMETERS.FILE_EXTENSION = PARAMETERS.update_file_extension(PARAMETERS)
                if i == 0:
                    df_train, df_test, y_train, y_test = load_datasets_for_lbp_operator(parent_path)
                else:
                    temp_datasets = load_datasets_for_lbp_operator(parent_path, discard_columns=True)
                    df_train = pd.concat([df_train, temp_datasets[0]], axis=1)
                    df_test = pd.concat([df_test, temp_datasets[1]], axis=1)
        else:
            df_train, df_test, y_train, y_test = load_datasets_for_lbp_operator(parent_path)

        if extra_features is not None:
            df_train = pd.concat([df_train, extra_features['train']], axis=1)
            df_test = pd.concat([df_test, extra_features['test']], axis=1)

        if df_train.shape[1] > 0:
            clf = init_clf_and_fit(df_train, y_train, lgb)
            y_predicted = clf.predict(df_test)
        else:
            flag = False

    if PARAMETERS.PLOT and flag:
        label_predicted = np.array(y_predicted)
        preprocess = Preprocess(height=608, width=576)
        images_path = f'{parent_path}/dataset/training/images/'
        images = sorted(os.listdir(images_path))[14:]
        masks_path = f'{parent_path}/dataset/training/mask/'
        masks = sorted(os.listdir(masks_path))[14:]
        for image_path, mask_path in zip(images, masks):
            img = preprocess.read_img(images_path + image_path).ravel()
            mask = preprocess.read_img(masks_path + mask_path)
            img = img[mask.ravel() > 100]
            preprocess.plot_preprocess_with_label(img, label_predicted[:len(img)], mask)
            label_predicted = np.delete(label_predicted, np.arange(len(img)))
            if plot_once:
                break

    if flag:
        acc = accuracy_score(y_test, y_predicted)
        f1 = f1_score(y_test, y_predicted)
        mat = confusion_matrix(y_test, y_predicted).ravel()
        if 'GRID_SEARCH' not in os.environ or os.environ['GRID_SEARCH'] != 'TRUE':
            print('Accuracy score: ' + str(acc) + '\n')
            print('F1 score: ' + str(f1) + '\n')
            print('Confusion matrix:\n')
            print_confusion_matrix(y_test, y_predicted)
            print(f'Sensivity: {int(mat[3]) / (int(mat[3]) + int(mat[2]))}')
            print(f'Specificity: {int(mat[0]) / (int(mat[0]) + int(mat[1]))}')
        return round(acc, 3), round(f1, 3), int(mat[0]), int(mat[1]), int(mat[2]), int(mat[3])
    else:
        return -1, -1, -1, -1, -1, -1


if __name__ == '__main__':
    if 'GRID_SEARCH' not in os.environ or os.environ['GRID_SEARCH'] != 'TRUE':
        main(lgb='Num')
    else:
        metrics = pd.DataFrame(columns=[
                    'LBP', 'Method', 'Interpolation', 'Balance', 'n_scales', 'x2', 'Gray Intensity',
                    'Accuracy', 'F1 score', 'tn', 'fp', 'fn', 'tp'
                ])
        main_path = str(Path(os.path.dirname(os.path.abspath(__file__))).parent)
        for filename in Path(f"{main_path}/DB").glob('train_train_*'):
            PARAMETERS.FILE_EXTENSION = \
                str(filename).replace('train_train_', '').split('/')[-1].replace('.zip', '').replace('.pkl', '')
            properties = PARAMETERS.FILE_EXTENSION.replace(
                'get_pyramid_dataset', 'get-pyramid-dataset').replace(
                'get_datasets_by_scale', 'get-dataset-by-scale').split('_')
            # if len(properties) < 7:
            PARAMETERS.LBP_METHOD = properties[0]
            PARAMETERS.METHOD = properties[1].replace(
                'get-pyramid-dataset', 'get_pyramid_dataset').replace('get-dataset-by-scale', 'get_datasets_by_scale')
            PARAMETERS.INTERPOLATION_ALGORITHM = properties[2]
            PARAMETERS.BALANCE = properties[3].replace('balance-', '') == 'True'
            PARAMETERS.N_SCALES = int(properties[4].replace('scales-', ''))
            PARAMETERS.X2SCALE = properties[5].replace('x2-', '') == 'True'
            PARAMETERS.GRAY_INTENSITY = properties[6].replace('gray-intensity-', '') == 'True'
            metrics = metrics.append(pd.DataFrame(
                (
                    PARAMETERS.LBP_METHOD,
                    PARAMETERS.METHOD,
                    PARAMETERS.INTERPOLATION_ALGORITHM,
                    PARAMETERS.BALANCE,
                    PARAMETERS.N_SCALES,
                    PARAMETERS.X2SCALE,
                    PARAMETERS.GRAY_INTENSITY
                ) + main(),
                index=[
                    'LBP', 'Method', 'Interpolation', 'Balance', 'n_scales', 'x2', 'Gray Intensity',
                    'Accuracy', 'F1 score', 'tn', 'fp', 'fn', 'tp'
                ]
            ).T, ignore_index=True)
        metrics.to_csv(f"{main_path}/Results/metrics.csv")
