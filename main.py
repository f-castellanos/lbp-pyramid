##
import os
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier

from confusion_matrix_pretty_print import print_confusion_matrix
from preprocess.preprocess import Preprocess
import PARAMETERS


def init_clf_and_fit(df, y, xgb=False):
    if xgb:
        classifier = XGBClassifier(n_jobs=-1, objective='binary:logistic')
    else:
        classifier = MultinomialNB(fit_prior=True)
    classifier.fit(df, y)
    return classifier


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


def main(xgb=False, plot_once=False):
    # Database unzip
    parent_path = str(Path(os.path.dirname(os.path.abspath(__file__))).parent)
    train_file_name = f"{parent_path}/DB/train_train_{PARAMETERS.FILE_EXTENSION}"
    with zipfile.ZipFile(f'{train_file_name}.zip', 'r') as zip_ref:
        zip_ref.extractall(f'{parent_path}/DB/')
    test_file_name = f"{parent_path}/DB/train_test_{PARAMETERS.FILE_EXTENSION}"
    with zipfile.ZipFile(f'{test_file_name}.zip', 'r') as zip_ref:
        zip_ref.extractall(f'{parent_path}/DB/')

    flag = True
    if PARAMETERS.METHOD == 'get_datasets_by_scale':
        with open(f'{train_file_name}.pkl', 'rb') as f:
            df_train_list = pickle.load(f)
        os.remove(f'{train_file_name}.pkl')
        clf_list = [init_clf_and_fit(df_train.iloc[:, :-1], df_train.iloc[:, -1], xgb) for df_train in df_train_list]
        with open(f'{test_file_name}.pkl', 'rb') as f:
            df_test_list = pickle.load(f)
        os.remove(f'{test_file_name}.pkl')
        y_predicted_test_list = [ensemble_prediction(clf_list, dfs_test) for dfs_test in df_test_list]
        y_predicted = np.concatenate([y_predicted_test_pic[0] for y_predicted_test_pic in y_predicted_test_list])
        y_predicted = np.where(y_predicted > 0.5, 1, 0)
        y_test = np.concatenate([y_predicted_test_pic[1] for y_predicted_test_pic in y_predicted_test_list])

    else:
        df_train = pd.read_pickle(f'{train_file_name}.pkl')
        os.remove(f'{train_file_name}.pkl')
        df_test = pd.read_pickle(f'{test_file_name}.pkl')
        os.remove(f'{test_file_name}.pkl')
        y_train = df_train.loc[:, 'label']
        y_test = df_test.loc[:, 'label']
        df_train.drop(columns=['label'], inplace=True)
        df_test.drop(columns=['label'], inplace=True)

        if df_train.shape[1] > 0:
            clf = init_clf_and_fit(df_train, y_train, xgb)
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
        if 'GRID_SEARCH' not in os.environ or os.environ['GRID_SEARCH'] != 'TRUE':
            print('Accuracy score: ' + str(acc) + '\n')
            print('F1 score: ' + str(f1) + '\n')
            print('Confusion matrix:\n')
            print_confusion_matrix(y_test, y_predicted)
        mat = confusion_matrix(y_test, y_predicted).ravel()
        return round(acc, 3), round(f1, 3), int(mat[0]), int(mat[1]), int(mat[2]), int(mat[3])
    else:
        return -1, -1, -1, -1, -1, -1


if __name__ == '__main__':
    if 'GRID_SEARCH' not in os.environ or os.environ['GRID_SEARCH'] != 'TRUE':
        main()
    else:
        metrics = pd.DataFrame(columns=[
                    'LBP', 'Method', 'Interpolation', 'Balance', 'n_scales', 'x2', 'Gray Intensity',
                    'Accuracy', 'F1 score', 'tn', 'fp', 'fn', 'tp'
                ])
        parent_path = str(Path(os.path.dirname(os.path.abspath(__file__))).parent)
        for filename in Path(f"{parent_path}/DB").glob('train_train_*'):
            PARAMETERS.FILE_EXTENSION = str(filename).replace('train_train_', '').split('/')[-1].replace('.zip', '').replace('.pkl', '')
            properties = PARAMETERS.FILE_EXTENSION.replace(
                'get_pyramid_dataset', 'get-pyramid-dataset').replace(
                'get_datasets_by_scale', 'get-dataset-by-scale').split('_')
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
        metrics.to_csv(f"{parent_path}/Results/metrics.csv")
