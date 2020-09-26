from preprocess.preproces import Preprocess
from os.path import dirname, abspath
from pathlib import Path
from preprocess.lbp import lbp
import pandas as pd
import numpy as np


lbp_img = lbp(str(Path(dirname(abspath(__file__))).parent) + '/dataset/training/images/21_training.tif')
lbp_img = pd.Series(lbp_img.reshape(1, -1)[0]).to_frame()
# lbp_img = np.array(lbp_img).reshape(584, -1)
