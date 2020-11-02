# Copyright 2020 Alexey Alexandrov <sks2311211@yandex.ru>

import genetic_algorithm as genetic
import numpy as np

if __name__ == '__main__':
    fit_function = lambda x, y: np.exp(- x ** 2) * np.exp(- y ** 2) / (1 + x ** 2 + y ** 2)
    bounds = [-2., 2., -2., 2.]

    gen_alg = genetic.GeneticAlgorithm(fit_function, bounds)
    gen_alg.PrintPopulation()
    gen_alg.Iterating()
