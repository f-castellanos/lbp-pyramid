from preprocess.preprocess import Preprocess
from plot_results.plot_results import PlotResults
import os
from pathlib import Path
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix


# Database load
parent_path = str(Path(os.path.dirname(os.path.abspath(__file__))).parent)
df_train = pd.read_pickle(parent_path + '/DDBB/train_train.pkl')
df_test = pd.read_pickle(parent_path + '/DDBB/train_test.pkl')
y_train = df_train.loc[:, 'label']
y_test = df_test.loc[:, 'label']
df_train.drop(columns=['label'], inplace=True)
df_test.drop(columns=['label'], inplace=True)

# Binning
bins = list(range(-1, 256, 32))
for column in df_train.columns:
    df_train.loc[:, column] = pd.cut(df_train.loc[:, column], bins, labels=np.array(bins[1:]).astype(str))
    df_test.loc[:, column] = pd.cut(df_test.loc[:, column], bins, labels=np.array(bins[1:]).astype(str))

clf = GaussianNB()
clf.fit(df_train, y_train)
y_pred = clf.predict(df_test)

print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
