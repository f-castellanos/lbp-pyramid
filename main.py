##
import os
from pathlib import Path
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import pickle
import numpy as np
from preprocess.preprocess import Preprocess


# LBP_METHOD = 'default'
# LBP_METHOD = 'riu'
LBP_METHOD = 'riu2'
# METHOD = 'get_pyramid_dataset'
METHOD = 'get_datasets_by_scale'
HEIGHT = 608


def init_clf_and_fit(df, y):
    classifier = GaussianNB()
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
    # Database load
    parent_path = str(Path(os.path.dirname(os.path.abspath(__file__))).parent)
    if METHOD == 'get_datasets_by_scale':
        with open(parent_path + '/DB/train_train_' + LBP_METHOD + '_' + METHOD + '.pkl', 'rb') as f:
            df_train_list = pickle.load(f)
        clf_list = [init_clf_and_fit(df_train.iloc[:, :-1], df_train.iloc[:, -1]) for df_train in df_train_list]
        with open(parent_path + '/DB/train_test_' + LBP_METHOD + '_' + METHOD + '.pkl', 'rb') as f:
            df_test_list = pickle.load(f)
        y_predicted_test_list = [ensemble_prediction(clf_list, dfs_test) for dfs_test in df_test_list]
        y_predicted = np.concatenate([y_predicted_test_pic[0] for y_predicted_test_pic in y_predicted_test_list])
        y_predicted = np.where(y_predicted > 0.5, 1, 0)
        y_test = np.concatenate([y_predicted_test_pic[1] for y_predicted_test_pic in y_predicted_test_list])
        print(accuracy_score(y_test, y_predicted))
        print(f1_score(y_test, y_predicted))
        print(confusion_matrix(y_test, y_predicted))

    elif METHOD == 'get_pyramid_dataset':
        df_train = pd.read_pickle(parent_path + '/DB/train_train_' + LBP_METHOD + '_' + METHOD + '.pkl')
        df_test = pd.read_pickle(parent_path + '/DB/train_test_' + LBP_METHOD + '_' + METHOD + '.pkl')
        y_train = df_train.loc[:, 'label']
        y_test = df_test.loc[:, 'label']
        df_train.drop(columns=['label'], inplace=True)
        df_test.drop(columns=['label'], inplace=True)

        clf = init_clf_and_fit(df_train, y_train)
        y_predicted = clf.predict(df_test)

        print(accuracy_score(y_test, y_predicted))
        print(f1_score(y_test, y_predicted))
        print(confusion_matrix(y_test, y_predicted))
