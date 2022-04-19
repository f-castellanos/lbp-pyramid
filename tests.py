import PARAMETERS
import numpy as np
import pickle


def convolution_features(kernel):
    from create_db import main
    PARAMETERS.METHOD = 'get_pyramid_dataset'
    PARAMETERS.INTERPOLATION_ALGORITHM = 'lanczos'
    PARAMETERS.BALANCE = False
    PARAMETERS.N_SCALES = 5
    PARAMETERS.GRAY_INTENSITY = True
    PARAMETERS.X2SCALE = True
    PARAMETERS.FILE_EXTENSION = PARAMETERS.update_file_extension(PARAMETERS)
    PARAMETERS.PLOT = False
    PARAMETERS.CONVOLUTION = kernel
    PARAMETERS.CONV_PATH = PARAMETERS.update_convolution_path(PARAMETERS)
    PARAMETERS.DATASET = 'DRIVE'
    PARAMETERS.CHANNEL = 1
    main()
    return f"../DB/{PARAMETERS.DATASET}/extra_features/convolution/{PARAMETERS.CONV_PATH}"


# def generate_features(kernel_list):
#     return Parallel(n_jobs=3)(delayed(convolution_features)(kernel) for kernel in kernel_list)

with open(
        r'/home/fer/Drive/Estudios/Master-IA/TFM/lbp-pyramid/convolution/outputs/16497076349499857 - 3LBP/population.pkl',
        'rb') as f:
    kernels = pickle.load(f)

population, fitness = kernels['population'], kernels['fitness']
individual = population[np.argmax(fitness), :]

kernel_list = []
count = 0
n_kernels = 3
k_size = (3, 5, 7)
for j, ks in enumerate(k_size):
    k_len = int(ks**2)
    kernel_list += [
        individual[(count + i*k_len):(count + (i + 1) * k_len)].reshape((ks, ks))
        for i in range(n_kernels // len(k_size))
    ]
    count += k_len * (n_kernels // len(k_size))


paths = convolution_features(kernel_list[0])
