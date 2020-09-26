from preprocess.preprocess import Preprocess
from os.path import dirname, abspath
from pathlib import Path
from preprocess.lbp import lbp
import pandas as pd
import numpy as np


preprocess = Preprocess(height=608, width=576)
path = str(Path(dirname(abspath(__file__))).parent) + '/dataset/training/images/21_training.tif'
mask_path = str(Path(dirname(abspath(__file__))).parent) + '/dataset/training/mask/21_training_mask.tif'
preprocess.apply_lbp(path, mask_path=mask_path)
