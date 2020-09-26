from preprocess.preprocess import Preprocess
from plot_results.plot_results import PlotResults
from os.path import dirname, abspath
from pathlib import Path
from preprocess.lbp import lbp
import pandas as pd
import numpy as np

parent_path = str(Path(dirname(abspath(__file__))).parent)
preprocess = Preprocess(height=608, width=576)
plot_results = PlotResults(height=608, width=576)
path = parent_path + '/dataset/training/images/21_training.tif'
mask_path = parent_path + '/dataset/training/mask/21_training_mask.tif'
df = preprocess.apply_lbp(path, mask_path=mask_path)
df.columns = ['1:1']
label_path = parent_path + '/dataset/training/1st_manual/21_manual1.gif'
df['target'] = preprocess.get_label(label_path)
plot_results.plot_label(df)
