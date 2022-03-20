import numpy as np
from fitness import fitness_function
import matplotlib.pyplot as plt
from pathlib import Path
import time
from joblib import Parallel, delayed
import math
import pickle

np.random.seed(1)
MIN_MAX = {'min': np.min, 'max': np.max}
MIN_MAX_R = {'min': np.max, 'max': np.min}
ARG_MIN_MAX = {'min': np.argmin, 'max': np.argmax}
ARG_MIN_MAX_R = {'min': np.argmax, 'max': np.argmin}


class EvolutionaryKernelOptimization:
    def __init__(self, elitism=True, p_size=70, p_n_kernels=2, k_size=3, method='max'):
    # def __init__(self, elitism=True, p_size=70, p_n_kernels=2, k_size=1, method='min'):  # benchmark function # noqa
        self.p_size = p_size
        self.p_n_kernels = p_n_kernels
        self.k_size = k_size
        self.elitism = elitism

        self.k = 3
        self.alpha = .6
        self.sigma = .4
        self.recombination_proba = .2
        self.mutation_proba = .05

        self.population = None
        self.fitness = None
        self.offspring = None
        self.offspring_fitness = None

        self.graph_fitness_best_individual = []
        self.graph_average_fitness = []
        self.graph_std = []

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
            self.population = np.random.uniform(-1, 1, (self.p_size, self.p_n_kernels*(self.k_size**2)))
            # self.population[:, 0] = self.population[:, 0] * 2  # benchmark function
            self.fitness = np.ones((self.p_size,)) * -1
            self.update_fitness()

    def optimize(self, iterations, plot=True, save_results=True):
        """
        Evolutionary algorithm main process. Carries out optimization.
        :param iterations: number of iterations of the optimization process.
        :param plot: whether to plot the optimization process.
        :param save_results: whether to save the results of the optimization process.
        """
        for _ in range(iterations):
            parent_indexes = self.tournament_selection()
            self.arithmetic_recombination(parent_indexes)
            self.uncorrelated_mutation()
            self.update_fitness(offspring=True)
            self.update_population()

            self.graph_fitness_best_individual.append(MIN_MAX[self.method](self.fitness))
            self.graph_average_fitness.append(np.mean(self.fitness))
            self.graph_std.append(np.std(self.fitness))
        if plot:
            plt.figure(figsize=(18, 10))
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
        if save_results:
            path = f"outputs/{time.time()}".replace('.', '')
            Path(path).mkdir(parents=True, exist_ok=True)
            if plot:
                plt.savefig(f'{path}/fitness.png')
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

    def subset_fitness(self, population, fitness):
        if (fitness == -1).any():
            fitness[fitness == -1] = np.apply_along_axis(
                fitness_function,
                1,
                population[fitness == -1, :],
                n_kernels=self.p_n_kernels
            )
        return fitness

    def update_fitness(self, offspring=False):
        """
        Fitness calculation for the new individuals of the population
        """
        p, f = (self.offspring, self.offspring_fitness) if offspring else (self.population, self.fitness)
        subset = f == -1
        step = math.ceil(subset.sum() / self.n_jobs)
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
            self.offspring = np.clip(self.offspring, -1, 1)
            # self.offspring[:, 0] = np.clip(self.offspring[:, 0], -2, 2)  # benchmark function
            # self.offspring[:, 1] = np.clip(self.offspring[:, 1], -1, 1)  # benchmark function
            self.offspring_fitness[mutation_selection.sum(axis=1).astype(bool)] = -1

    def update_population(self):
        """
        Replaces the population by offspring. Elitism allows to preserve the best individual of the population.
        """
        if self.elitism and MIN_MAX[self.method](self.fitness) > MIN_MAX_R[self.method](self.offspring_fitness):
            best_individual = ARG_MIN_MAX[self.method](self.fitness)
            worst_individual = ARG_MIN_MAX_R[self.method](self.offspring_fitness)
            self.offspring[worst_individual, :] = self.population[best_individual, :]
            offspring_fitness = list(self.offspring_fitness)
            offspring_fitness[worst_individual] = self.fitness[best_individual]
            self.offspring_fitness = np.array(offspring_fitness)
        self.population = self.offspring
        self.fitness = self.offspring_fitness


dev = EvolutionaryKernelOptimization()
dev.optimize(iterations=300)
