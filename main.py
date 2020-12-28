##
import os
from pathlib import Path
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder
import pickle
import numpy as np
from preprocess.preprocess import Preprocess
from confusion_matrix_pretty_print import print_confusion_matrix
import zipfile


PLOT = True
# LBP_METHOD = 'default'
# LBP_METHOD = 'riu'
LBP_METHOD = 'riu2'
METHOD = 'get_pyramid_dataset'
# METHOD = 'get_datasets_by_scale'
HEIGHT = 608


def init_clf_and_fit(df, y):
    classifier = MultinomialNB(fit_prior=True)
    classifier.fit(df, y)
    return classifier


def ensemble_prediction(classifiers, dfs_test):
    # Calculating predictions
    y_predicted_list = [classifier.predict(test_set.iloc[:, :-1]).reshape(-1, 1)
                        for classifier, test_set in zip(classifiers, dfs_test['datasets'])]
    # Equaling the number of pixels
    y_predicted_list = [y_predicted_list[i].reshape(HEIGHT//(2 ** i), -1) for i in range(len(y_predicted_list))]
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


if __name__ == '__main__':
    # Database unzip
    parent_path = str(Path(os.path.dirname(os.path.abspath(__file__))).parent)
    train_file_name = parent_path + '/DB/train_train_' + LBP_METHOD + '_' + METHOD
    with zipfile.ZipFile(f'{train_file_name}.zip', 'r') as zip_ref:
        zip_ref.extractall(f'{parent_path}/DB/')
    test_file_name = parent_path + '/DB/train_test_' + LBP_METHOD + '_' + METHOD
    with zipfile.ZipFile(f'{test_file_name}.zip', 'r') as zip_ref:
        zip_ref.extractall(f'{parent_path}/DB/')

    if METHOD == 'get_datasets_by_scale':
        with open(f'{train_file_name}.pkl', 'rb') as f:
            df_train_list = pickle.load(f)
        os.remove(f'{train_file_name}.pkl')
        clf_list = [init_clf_and_fit(df_train.iloc[:, :-1], df_train.iloc[:, -1]) for df_train in df_train_list]
        with open(f'{test_file_name}.pkl', 'rb') as f:
            df_test_list = pickle.load(f)
        os.remove(f'{test_file_name}.pkl')
        y_predicted_test_list = [ensemble_prediction(clf_list, dfs_test) for dfs_test in df_test_list]
        y_predicted = np.concatenate([y_predicted_test_pic[0] for y_predicted_test_pic in y_predicted_test_list])
        y_predicted = np.where(y_predicted > 0.5, 1, 0)
        y_test = np.concatenate([y_predicted_test_pic[1] for y_predicted_test_pic in y_predicted_test_list])

        print('Accuracy score: ' + str(accuracy_score(y_test, y_predicted)) + '\n')
        print('F1 score: ' + str(f1_score(y_test, y_predicted)) + '\n')
        print('Confusion matrix:\n')
        print_confusion_matrix(y_test, y_predicted)

    else:
        df_train = pd.read_pickle(f'{train_file_name}.pkl')
        os.remove(f'{train_file_name}.pkl')
        df_test = pd.read_pickle(f'{test_file_name}.pkl')
        os.remove(f'{test_file_name}.pkl')
        y_train = df_train.loc[:, 'label']
        y_test = df_test.loc[:, 'label']
        df_train.drop(columns=['label'], inplace=True)
        df_test.drop(columns=['label'], inplace=True)

        clf = init_clf_and_fit(df_train, y_train)
        y_predicted = clf.predict(df_test)

        print('Accuracy score: ' + str(accuracy_score(y_test, y_predicted)) + '\n')
        print('F1 score: ' + str(f1_score(y_test, y_predicted)) + '\n')
        print('Confusion matrix:\n')
        print_confusion_matrix(y_test, y_predicted)

    if PLOT:
        label_predicted = np.array(y_predicted)
        preprocess = Preprocess(height=608, width=576)
        images_path = f'{parent_path}/dataset/training/images/'
        images = sorted(os.listdir(images_path))[14:]
        masks_path = f'{parent_path}/dataset/training/mask/'
        masks = sorted(os.listdir(masks_path))[14:]
        for image_path, mask_path in zip(images, masks):
            img = Preprocess.read_img(images_path + image_path).ravel()
            mask = Preprocess.read_img(masks_path + mask_path)
            img = img[mask.ravel() > 100]
            preprocess.plot_preprocess_with_label(img, label_predicted[:len(img)], mask)
            label_predicted = np.delete(label_predicted, np.arange(len(img)))
