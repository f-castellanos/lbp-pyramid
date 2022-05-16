from convolution.kernel_optimization import EvolutionaryKernelOptimization


kwargs = {
    'p_size': 50,
    'p_n_kernels': 8,
    'k_size': 1,
    'method': 'max',
    'clip_v': 100,
    'k': 2,
    'mutation_proba': .1,
    'recombination_proba': .6,
    'n_jobs': 5,
    'apply_abs': True
}

dev = EvolutionaryKernelOptimization(**kwargs)
# dev.fitness_function = 'PREPROCESS'
dev.fitness_function = 'PREPROCESS_W'
dev.init_population()
# dev.optimize(iterations=5, plot='lineal', save_results=True)
dev.optimize(iterations=150, plot='lineal', save_results=True)
