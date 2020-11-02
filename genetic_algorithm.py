# Copyright 2020 Alexey Alexandrov <sks2311211@yandex.ru>

import random

import numpy as np
from prettytable import PrettyTable

ITERATIONS_COUNT = 100
SPECIES_COUNT = 4
MUTATE_CHANCE = 0.25


class GeneticAlgorithm:

    def __init__(self, fit_func, bounds):
        self.fit_func = fit_func
        self.iterations_count = ITERATIONS_COUNT
        self.bounds = bounds
        self.species_count = SPECIES_COUNT
        self.population_number = 0
        self.population = self.InitPopulation()

        self.max_result = np.max(self.population[:, 2])
        self.mean_result = np.mean(self.population[:, 2])

    def PrintPopulation(self):
        print(f"№ поколения: {self.population_number}")
        table = PrettyTable()
        table.field_names = ["X", "Y", "FIT"]

        for i in range(self.species_count):
            table.add_row(list(self.population[i]))

        print(table)
        self.population_number += 1

        print(f"Максимальный результат: {self.max_result}")
        print(f"Средний результат: {self.mean_result}")

    def Iterating(self):
        for i in range(self.iterations_count):
            self.Iterate()
            self.PrintPopulation()

    def InitPopulation(self):
        population = np.random.rand(self.species_count, 3)

        population[:, 0] = self.bounds[0] + population[:, 0] * (self.bounds[1] - self.bounds[0])
        population[:, 1] = self.bounds[2] + population[:, 1] * (self.bounds[3] - self.bounds[2])

        population[:, 2] = self.fit_func(population[:, 0], population[:, 1])

        return population

    def Iterate(self):
        curr_population = self.population[self.population[:, 2].argsort()].copy()

        # Mutate in 25% of population.
        if random.uniform(0, 1) <= 0.25:
            delta = (np.random.rand(*self.population.shape) - 0.5) / 10
            curr_population = curr_population + delta

        x3 = curr_population[3][0]
        y3 = curr_population[3][1]
        x2 = curr_population[2][0]
        y2 = curr_population[2][1]
        x1 = curr_population[1][0]
        y1 = curr_population[1][1]

        new_population = np.array(
            [[x3, y1, self.fit_func(x3, y1)], [x1, y3, self.fit_func(x1, y3)],
             [x3, y2, self.fit_func(x3, y2)], [x2, y3, self.fit_func(x2, y3)]])

        self.max_result = np.max(new_population[:, 2])
        self.mean_result = np.mean(new_population[:, 2])

        self.population = new_population.copy()
