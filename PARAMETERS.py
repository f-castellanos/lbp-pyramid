"""
LBP_METHOD
----------
Defines the LBP algorithm version to be used.
· default: LBP original version [https://medium.com/@rajatanantharam/local-binary-patterns-8807ecf7f87c].
· riu: Rotation invariant version [https://www.researchgate.net/publication/221303862_Gray_Scale_and_Rotation_Invariant_Texture_Classification_with_Local_Binary_Patterns].
· riu2: Improved rotation invariance with uniform patterns [https://www.researchgate.net/publication/221303862_Gray_Scale_and_Rotation_Invariant_Texture_Classification_with_Local_Binary_Patterns].
"""  # noqa
LBP_METHOD = 'riu2'

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
INTERPOLATION_ALGORITHM = 'nearest'

"""
BALANCE
-------
Whether to remove data from the train set in order to equalize the proportion of instances of each label.
· True
· False
"""
BALANCE = False

# Other parameters
PLOT = False
N_JOBS = 1
HEIGHT = 608
