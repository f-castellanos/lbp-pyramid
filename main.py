##
import os
from pathlib import Path
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix


if __name__ == '__main__':
    lbp_method = 'riu2'
    # Database load
    parent_path = str(Path(os.path.dirname(os.path.abspath(__file__))).parent)
    df_train = pd.read_pickle(parent_path + '/DB/train_train_' + lbp_method + '.pkl')
    df_test = pd.read_pickle(parent_path + '/DB/train_test_' + lbp_method + '.pkl')
    y_train = df_train.loc[:, 'label']
    y_test = df_test.loc[:, 'label']
    df_train.drop(columns=['label'], inplace=True)
    df_test.drop(columns=['label'], inplace=True)

    clf = GaussianNB()
    clf.fit(df_train, y_train)
    y_predicted = clf.predict(df_test)

    print(accuracy_score(y_test, y_predicted))
    print(f1_score(y_test, y_predicted))
    print(confusion_matrix(y_test, y_predicted))
