##
import os
from pathlib import Path
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import pickle
import numpy as np


# LBP_METHOD = 'default'
# LBP_METHOD = 'riu'
LBP_METHOD = 'riu2'
# METHOD = 'get_pyramid_dataset'
METHOD = 'get_datasets_by_scale'


def init_clf_and_fit(df, y):
    classifier = GaussianNB()
    classifier.fit(df, y)
    print(sum(y)/len(y))
    return classifier


def ensemble_prediction(classifiers, x_test_list):
    y_predicted_list = [classifier.predict(x_test).reshape(-1, 1)
                        for classifier, x_test in zip(classifiers, x_test_list)]
    # img_lbp = Preprocess.repeat_pixels(img_lbp, i)
    print(np.mean(np.concatenate(y_predicted_list, axis=1), axis=0))
    return np.mean(np.concatenate(y_predicted_list, axis=1), axis=1)


if __name__ == '__main__':
    # Database load
    parent_path = str(Path(os.path.dirname(os.path.abspath(__file__))).parent)
    if METHOD == 'get_datasets_by_scale':
        with open(parent_path + '/DB/train_train_' + LBP_METHOD + '_' + METHOD + '.pkl', 'rb') as f:
            df_train_list = pickle.load(f)
        clf_list = [init_clf_and_fit(df_train.iloc[:, :-1], df_train.iloc[:, -1]) for df_train in df_train_list]
        with open(parent_path + '/DB/train_test_' + LBP_METHOD + '_' + METHOD + '.pkl', 'rb') as f:
            df_test_list = pickle.load(f)
        x_test_set = [df_test.iloc[:, :-1] for df_test in df_test_list]
        y_predicted = ensemble_prediction(clf_list, x_test_set)

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
