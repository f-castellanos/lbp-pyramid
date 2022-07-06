from convolution.kernel_optimization import EvolutionaryKernelOptimization


kwargs = {
    'p_size': 50,
    'p_n_kernels': 8,
    'k_size': 1,
    'method': 'max',
    'clip_v': 100,
    'k': 2,
    'mutation_proba': 1.5,
    'recombination_proba': .6,
    # 'n_jobs': 1,
    'n_jobs': 10,
    'apply_abs': True
}

dev = EvolutionaryKernelOptimization(**kwargs)
# dev.fitness_function = 'PREPROCESS'
dev.fitness_function = 'PREPROCESS_LBP_G_CV'
# dev.fitness_function = 'PREPROCESS_LBP_G'
# dev.fitness_function = 'PREPROCESS_LBP_GB'
# dev.fitness_function = 'PREPROCESS_GB'
# dev.fitness_function = 'PREPROCESS_W'
# dev.fitness_function = 'PREPROCESS_LBP'
dev.init_population()
# dev.optimize(iterations=5, plot='lineal', save_results=True)
dev.optimize(iterations=70, plot='lineal', save_results=True)
# dev.optimize(iterations=100, plot='lineal', save_results=True)
