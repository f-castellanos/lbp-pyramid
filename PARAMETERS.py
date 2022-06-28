import numpy as np
import hashlib

"""
LBP_METHOD
----------
Defines the LBP algorithm version to be used.
· default: LBP original version [https://medium.com/@rajatanantharam/local-binary-patterns-8807ecf7f87c].
· riu: Rotation invariant version [https://www.researchgate.net/publication/221303862_Gray_Scale_and_Rotation_Invariant_Texture_Classification_with_Local_Binary_Patterns].
· riu2: Improved rotation invariance with uniform patterns [https://www.researchgate.net/publication/221303862_Gray_Scale_and_Rotation_Invariant_Texture_Classification_with_Local_Binary_Patterns].
· nriuniform: non rotation-invariant uniform patterns variant which is only gray scale invariant
· var: rotation invariant variance measures of the contrast of local image texture which is rotation but not gray scale invariant.

"""  # noqa
LBP_METHOD = 'var'
# LBP_METHOD = 'riu'
# LBP_METHOD = 'default'

"""
METHOD
------
Defines the dataset structure.
· get_pyramid_dataset: Generate a single dataset in which each variable is a scale of the original image. Therefore, a single classifier is used.
· get_datasets_by_scale: Generate a dataset for each scale. Therefore, a different classifier is applied for scale.
"""  # noqa
METHOD = 'get_pyramid_dataset'

"""
INTERPOLATION_ALGORITHM
-----------------------
Defines the interpolation algorithm to be used in the image rescaling process.
· nearest: Pick one nearest pixel from the input image. Ignore all other input pixels.
· bicubic: Cubic interpolation on all pixels that may contribute to the output value.
· lanczos: Calculate the output pixel value using a high-quality Lanczos filter.
"""
# INTERPOLATION_ALGORITHM = 'nearest'
INTERPOLATION_ALGORITHM = 'lanczos'

"""
BALANCE    <- ONLY for get_pyramid_dataset
-------
Whether to remove data from the train set in order to equalize the proportion of instances of each label.
· True
· False
"""
BALANCE = False
# BALANCE = False

"""
N_SCALES
--------
Number of scales to use.
1 - 6
"""
# N_SCALES = 1
N_SCALES = 5

"""
GRAY_INTENSITY    <- ONLY for get_pyramid_dataset
--------------
Whether to add a variable with the intensity of each pixel in the gray scale image.
· True
· False
"""
# GRAY_INTENSITY = False
GRAY_INTENSITY = True

"""
X2SCALE
-------
Whether to add a scale with x2 resolution.
· True
· False
"""
X2SCALE = True
# X2SCALE = False

"""
"""
ENCODING = 'categorical'

CONVOLUTION = None
PREPROCESS_OPTIMIZATION = True
# PREPROCESS_TYPE = 'default'
PREPROCESS_TYPE = 'lbp_gb'
# CONVOLUTION = np.round(np.random.uniform(low=-1, high=1, size=(9,)).reshape(3, 3), 3)
CONV_PATH = None if CONVOLUTION is None else ';'.join(CONVOLUTION.ravel().astype(str))
CONV_PREPROCESSING = False
RADIUS = 1


CHANNEL = 2
# CHANNEL = None

# Other parameters
MODEL_NAME = ''
PLOT = False
PLOT_LBP_LABEL = False
N_JOBS = 1
# HEIGHT = 608
# WIDTH = 576
FILE_EXTENSION = f"{LBP_METHOD}_{METHOD}_{INTERPOLATION_ALGORITHM}" \
                 f"_balance-{BALANCE}_scales-{N_SCALES}_x2-{X2SCALE}" \
                 f"_gray-intensity-{GRAY_INTENSITY}"
# FILE_EXTENSION = f"{LBP_METHOD}_{METHOD}_{INTERPOLATION_ALGORITHM}" \
#                  f"_balance-{BALANCE}_scales-{N_SCALES}_x2-{X2SCALE}" \
#                  f"_gray-intensity-{GRAY_INTENSITY}"

DATASET = 'STARE'


def update_file_extension(parameters):
    return f"{parameters.LBP_METHOD}_{parameters.METHOD}_{parameters.INTERPOLATION_ALGORITHM}" \
                 f"_balance-{parameters.BALANCE}_scales-{parameters.N_SCALES}_x2-{parameters.X2SCALE}" \
                 f"_gray-intensity-{parameters.GRAY_INTENSITY}"


def update_convolution_path(parameters):
    return hashlib.sha256(';'.join(parameters.CONVOLUTION.ravel().astype(str)).encode('utf-8')).hexdigest()
    # return ';'.join(parameters.CONVOLUTION.ravel().astype(str))
