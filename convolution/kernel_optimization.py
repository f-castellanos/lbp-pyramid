import numpy as np
from fitness import fitness_function
import matplotlib.pyplot as plt
from pathlib import Path
import time
from joblib import Parallel, delayed
import math
import pickle
from tqdm import tqdm

np.random.seed(1)
FITNESS_FUNCTION = 'F1'
MIN_MAX = {'min': np.min, 'max': np.max}
MIN_MAX_R = {'min': np.max, 'max': np.min}
ARG_MIN_MAX = {'min': np.argmin, 'max': np.argmax}
ARG_MIN_MAX_R = {'min': np.argmax, 'max': np.argmin}


def elitism_comparison(v1, v2, method):
    if method == 'max':
        return v1 > v2
    else:
        return v1 < v2


class EvolutionaryKernelOptimization:
    def __init__(self, elitism=True, p_size=70, p_n_kernels=2, k_size=3,
                 method='max', clip_v=1, k=3, mutation_proba=.05, recombination_proba=.2):
        self.p_size = p_size
        self.p_n_kernels = p_n_kernels
        if isinstance(k_size, int):
            k_size = (k_size,)
        self.k_size = np.array(k_size)
        self.elitism = elitism
        self.clip_v = clip_v

        self.k = k
        self.alpha = .6
        self.sigma = .4
        self.recombination_proba = recombination_proba
        self.mutation_proba = mutation_proba
        # self.k = 3
        # self.alpha = .6
        # self.sigma = 1
        # self.recombination_proba = .8
        # self.mutation_proba = .5

        self.population = None
        self.fitness = None
        self.offspring = None
        self.offspring_fitness = None

        self.graph_fitness_best_individual = []
        self.graph_average_fitness = []
        self.graph_std = []

        # self.n_jobs = 1
        self.n_jobs = 6
        self.method = method

        self.init_population()
        # import matplotlib.pyplot as plt
        # from PIL import Image
        # from fitness import IMAGES
        # im = Image.fromarray(np.uint8(IMAGES[0]))
        # plt.figure(figsize=(15, 11), dpi=80)
        # plt.imshow(im, cmap='gray')
        # plt.show()
        # a = 0

    def init_population(self):
        """
        Initializes the population
        """
        if self.population is None:
            self.population = np.random.uniform(
                -1, 1, (self.p_size, (self.p_n_kernels // len(self.k_size))*(np.sum(self.k_size**2))))
            if self.clip_v != 1:
                self.population = self.population * self.clip_v
            # self.population[:, 0] = self.population[:, 0] * 2  # benchmark function
            self.fitness = np.ones((self.p_size,)) * -1
            self.update_fitness()

    def optimize(self, iterations, plot='log', save_results=True):
        """
        Evolutionary algorithm main process. Carries out optimization.
        :param iterations: number of iterations of the optimization process.
        :param plot: whether to plot the optimization process.
        :param save_results: whether to save the results of the optimization process.
        """
        for _ in tqdm(range(iterations)):
            parent_indexes = self.tournament_selection()
            self.arithmetic_recombination(parent_indexes)
            self.uncorrelated_mutation()
            self.update_fitness(offspring=True)
            self.update_population()

            self.graph_fitness_best_individual.append(MIN_MAX[self.method](self.fitness))
            self.graph_average_fitness.append(np.mean(self.fitness))
            self.graph_std.append(np.std(self.fitness))
        if save_results:
            path = f"outputs/{time.time()}".replace('.', '')
            Path(path).mkdir(parents=True, exist_ok=True)
            text = f"""
PARAMETERS
----------
Population size: {self.p_size}
k (tournament selection): {self.k}
alpha (recombination): {self.alpha}
sigma (mutation): {self.sigma}

Mutation probability: {self.mutation_proba}
Recombination probability: {self.recombination_proba}

BEST RESULT
-----------
Fitness: {MIN_MAX[self.method](self.fitness)}
Kernels: {self.population[ARG_MIN_MAX[self.method](self.fitness), :]}

KERNELS
-------
            """
            for i in range(self.p_size):
                text += f"\n{round(self.fitness[i], 3)} - {self.population[i, :].tolist()}"
            with open(f"{path}/results.txt", 'w') as f:
                f.write(text)
            with open(f'{path}/population.pkl', 'wb') as handle:
                pickle.dump(
                    {'population': self.population, 'fitness': self.fitness}, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(f'{path}/backup.pkl', 'wb') as handle:
                pickle.dump(
                    [self.graph_fitness_best_individual, self.graph_average_fitness, self.graph_std],
                    handle, protocol=pickle.HIGHEST_PROTOCOL)

        plt.figure(figsize=(18, 10))
        if plot == 'log':
            plt.yscale('log', base=10)
            delta = 1 - np.min(self.graph_average_fitness)
            if delta > 0:
                self.graph_fitness_best_individual += delta
                self.graph_average_fitness += delta
        plt.plot(self.graph_fitness_best_individual, label='Best individual')
        plt.plot(self.graph_average_fitness, label='Average fitness')
        plt.fill_between(
            range(iterations),
            np.array(self.graph_average_fitness) - np.array(self.graph_std),
            np.array(self.graph_average_fitness) + np.array(self.graph_std),
            alpha=.1
        )
        plt.legend(loc="upper left")
        if not save_results:
            plt.show()
        else:
            plt.savefig(f'{path}/fitness.png')

    def subset_fitness(self, population, fitness):
        if (fitness == -1).any():
            fitness[fitness == -1] = np.apply_along_axis(
                fitness_function,
                1,
                population[fitness == -1, :],
                n_kernels=self.p_n_kernels,
                k_size=self.k_size,
                function_name=FITNESS_FUNCTION
            )
        return fitness

    def update_fitness(self, offspring=False):
        """
        Fitness calculation for the new individuals of the population
        """
        p, f = (self.offspring, self.offspring_fitness) if offspring else (self.population, self.fitness)
        subset = f == -1
        step = math.ceil(subset.sum() / self.n_jobs)
        if step > 0:
            updated_fitness = Parallel(n_jobs=self.n_jobs)(delayed(self.subset_fitness)(
                p[subset, :][i:i+step], f[subset][i:i+step]) for i in range(0, subset.sum(), step))
            updated_fitness = np.array([item for sublist in updated_fitness for item in sublist])
            if offspring:
                self.offspring_fitness[subset] = updated_fitness
            else:
                self.fitness[subset] = updated_fitness

    def tournament_selection(self):
        """
        Choice of the parent population of the same size of the original population.
        The choice is made by comparison of the fitness of a random selection of k individuals.
        self.k: number of individuals to be compared in the choice of a parent.
        :return: indexes of the selected parents.
        """
        choice = np.random.randint(self.p_size, size=(self.p_size, self.k))
        return choice[np.arange(self.p_size), ARG_MIN_MAX[self.method](self.fitness[choice], axis=1)]

    def arithmetic_recombination(self, indexes):
        """
        Provides the offspring population for the parent population given.
        :param indexes: indexes of the selected parents.
        self.recombination_proba: probability of recombination of two parents
        self.alpha: recombination parameter
        """
        offspring = []
        offspring_fitness = []
        for parents in indexes.reshape(self.p_size//2, 2):
            if np.random.rand() <= self.recombination_proba and (
                    self.population[parents[0]] != self.population[parents[1]]).any():
                offspring += [
                    self.alpha * self.population[parents[0]] + (1 - self.alpha) * self.population[parents[1]],
                    self.alpha * self.population[parents[1]] + (1 - self.alpha) * self.population[parents[0]],
                ]
                offspring_fitness += [-1, -1]
            else:
                offspring += [self.population[parents[0]], self.population[parents[1]]]
                offspring_fitness += [self.fitness[parents[0]], self.fitness[parents[1]]]
        self.offspring = np.array(offspring)
        self.offspring_fitness = np.array(offspring_fitness)

    def uncorrelated_mutation(self):
        """
        Mutation based on a normal distribution.
        self.mutation_proba: probability of an individual being mutated.
        self.sigma: normal distribution standard deviation.
        """
        mutation_selection = np.random.randint(0, 100, size=self.population.shape) < self.mutation_proba * 100
        if mutation_selection.any():
            self.offspring[mutation_selection] += np.random.normal(
                0, self.sigma, size=mutation_selection.sum()
            ) * self.offspring[mutation_selection]
            # self.offspring = np.clip(self.offspring, -self.clip_v, self.clip_v)
            # self.offspring[:, 0] = np.clip(self.offspring[:, 0], -2, 2)  # benchmark function
            # self.offspring[:, 1] = np.clip(self.offspring[:, 1], -1, 1)  # benchmark function
            self.offspring_fitness[mutation_selection.sum(axis=1).astype(bool)] = -1

    def update_population(self):
        """
        Replaces the population by offspring. Elitism allows to preserve the best individual of the population.
        """
        if self.elitism and elitism_comparison(
                MIN_MAX[self.method](self.fitness),
                MIN_MAX[self.method](self.offspring_fitness),
                self.method
        ):
            best_individual = ARG_MIN_MAX[self.method](self.fitness)
            worst_individual = ARG_MIN_MAX_R[self.method](self.offspring_fitness)
            self.offspring[worst_individual, :] = self.population[best_individual, :]
            offspring_fitness = list(self.offspring_fitness)
            offspring_fitness[worst_individual] = self.fitness[best_individual]
            self.offspring_fitness = np.array(offspring_fitness)
        self.population = self.offspring
        self.fitness = self.offspring_fitness


kwargs = {
    'SPHERE': {
        'p_size': 10,
        'p_n_kernels': 20,
        'k_size': 1,
        'method': 'min',
        'clip_v': 50
    },
    'RASTRIGIN': {
        'p_size': 10,
        'p_n_kernels': 5,
        'k_size': 1,
        'method': 'min',
        'clip_v': 5.12
    },
    # 'F1': {
    #     'p_size': 50,
    #     'p_n_kernels': 6,
    #     'k_size': (3, 5, 7),
    #     'method': 'max',
    #     'clip_v': 1
    # },
    'F1': {
        'p_size': 80,
        'p_n_kernels': 6,
        'k_size': (3, 5, 7),
        'method': 'max',
        'clip_v': 1,
        'k': 2,
        'mutation_proba': .1,
        'recombination_proba': .6
    }
}[FITNESS_FUNCTION]

dev = EvolutionaryKernelOptimization(**kwargs)
dev.optimize(iterations=150, plot='lineal', save_results=True)
# dev.optimize(iterations=1500, plot='log', save_results=True)
# dev.optimize(iterations=5000, plot='log', save_results=False)


#
# import numpy as np
# a = np.array([0.7229746483530723, 0.08515192263657567, 0.5182647054318817, -0.577948184085242, -0.9989541553663077, -0.46437588131189556, -0.015762936300529815, -0.35556680561587284, 1.0, -0.03396069276807806, 1.0, 0.03233413612776116, 0.15589639839299552, 0.790919536475375, 0.006393604211677433, -0.001971978111566875, -0.07670632347266587, -0.44206823803265893])
# k1 = a[:9].reshape(3, 3)
# k2 = a[9:].reshape(3, 3)
#
# PATH = r'/home/fer/Drive/Estudios/Master-IA/TFM/dataset/DRIVE/training/images'
# path = PATH + '/35_training.tif'
#
# from os import listdir
# import cv2
# import numpy as np
# import pandas as pd
# from PIL import Image
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import f1_score
# from preprocess.preprocess import Preprocess
#
# img = Preprocess.img_processing(np.asarray(Image.open(path).convert('RGB'))[:, :, 1])
# img_k1 = cv2.filter2D(img, -1, k1)
# img_k2 = cv2.filter2D(img, -1, k2)
# import matplotlib.pyplot as plt
# from PIL import Image
# from fitness import IMAGES
#
# im1 = Image.fromarray(np.uint8(img_k1))
# # plt.figure(figsize=(15, 11), dpi=80)
# # plt.imshow(im, cmap='gray')
# # plt.show()
#
# im2 = Image.fromarray(np.uint8(img_k2))
# # plt.figure(figsize=(15, 11), dpi=80)
# # plt.imshow(im, cmap='gray')
# # plt.show()
#
# fig = plt.figure()
#
# plt.subplot(1, 3, 1)
# plt.imshow(im1, cmap='gray')
#
# plt.subplot(1, 3, 2)
# plt.imshow(im2, cmap='gray')
#
# plt.subplot(1, 3, 3)
# plt.imshow(img, cmap='gray')
#
# # plt.subplot(1, 2, 2)
# # plt.imshow(im2, cmap='gray')
#
# plt.show()