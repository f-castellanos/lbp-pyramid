import numpy as np
import pickle
import PARAMETERS
# np.random.seed(1)
# folds = []
# img_list = np.arange(20)
# for _ in range(1):
#     np.random.shuffle(img_list)
#     folds.append(img_list.copy())
#
# for r_i in range(4, 20, 4):
#     folds.append(np.roll(folds[0], r_i))
with open(r"/tmp/folds.pkl", "wb") as output_file:
    pickle.dump([np.arange(28)], output_file)
with open(r"/tmp/index.pkl", "wb") as output_file:
    pickle.dump(0, output_file)

from convolution.kernel_optimization import EvolutionaryKernelOptimization


kwargs = {
    'p_size': 50,
    'p_n_kernels': 8,
    'k_size': 1,
    'method': 'max',
    'clip_v': 100,
    'k': 2,
    'mutation_proba': 0.15,
    'recombination_proba': .6,
    'n_jobs': 1,
    # 'n_jobs': 10,
    'apply_abs': True
}

# dev = EvolutionaryKernelOptimization(**kwargs)
# # # dev.fitness_function = 'PREPROCESS'
# # dev.fitness_function = 'PREPROCESS_LBP_G_FOLD'
# # # dev.fitness_function = 'PREPROCESS_LBP_G_CV'
# dev.fitness_function = 'PREPROCESS_LBP_G'
# # # dev.fitness_function = 'PREPROCESS_LBP_GB'
# # # dev.fitness_function = 'PREPROCESS_GB'
# # # dev.fitness_function = 'PREPROCESS_W'
# # # dev.fitness_function = 'PREPROCESS_LBP'
# dev.init_population()
# # # dev.optimize(iterations=5, plot='lineal', save_results=True)
# # dev.optimize(iterations=60, plot='lineal', save_results=True)
# dev.optimize(iterations=100, plot='lineal', save_results=True)

# for i_fold in range(5):
#     with open(r"/tmp/index.pkl", "wb") as output_file:
#         pickle.dump(i_fold, output_file)
#     print(folds[i_fold])
#     dev = EvolutionaryKernelOptimization(**kwargs)
#     dev.fitness_function = 'PREPROCESS_LBP_G_FOLD'
#     dev.init_population()
#     dev.optimize(iterations=100, plot='lineal', save_results=True)

for i_fold in range(PARAMETERS.FOLDS):
    PARAMETERS.CURRENT_FOLD = i_fold
    dev = EvolutionaryKernelOptimization(**kwargs)
    dev.fitness_function = 'PREPROCESS_LBP_G_FOLD'
    dev.init_population()
    dev.optimize(iterations=100, plot='lineal', save_results=True)


