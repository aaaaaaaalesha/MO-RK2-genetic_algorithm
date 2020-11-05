# Copyright 2020 Alexey Alexandrov <sks2311211@yandex.ru>

import random

import numpy as np
from prettytable import PrettyTable

ITERATIONS_COUNT = 20
SPECIES_COUNT = 4
MUTATE_CHANCE = 0.25
MUTATE_CONSTANT = 5


class GeneticAlgorithm:
    def __init__(self, fit_func, bounds):
        """Инициализация задачи по фит функции и границам диапазона."""
        self.fit_func_ = fit_func
        self.iterations_count_ = ITERATIONS_COUNT
        self.bounds_ = bounds
        self.species_count_ = SPECIES_COUNT
        self.population_number_ = 0
        self.population_ = self.InitPopulation()

        self.max_result_ = np.max(self.population_[:, 2])
        self.mean_result_ = np.mean(self.population_[:, 2])

    def PrintPopulation(self):
        """Вывод таблицы."""
        print(f"№ поколения: {self.population_number_}")
        table = PrettyTable()
        table.field_names = ["X", "Y", "FIT"]

        for i in range(self.species_count_):
            table.add_row(list(self.population_[i]))

        print(table)
        self.population_number_ += 1

        print(f"Максимальный результат: {self.max_result_}")
        print(f"Средний результат: {self.mean_result_}")

    def Solve(self):
        """Решение задачи."""
        for i in range(self.iterations_count_):
            # Mutate in 25% of population.
            if random.uniform(0, 1) <= 0.25:
                self.Mutation()
            self.CrossoverAndCalcFit()
            self.PrintPopulation()

    def InitPopulation(self):
        """Начальная популяция."""
        population = np.random.rand(self.species_count_, 3)

        population[:, 0] = self.bounds_[0] + population[:, 0] * (self.bounds_[1] - self.bounds_[0])
        population[:, 1] = self.bounds_[2] + population[:, 1] * (self.bounds_[3] - self.bounds_[2])

        population[:, 2] = self.fit_func_(population[:, 0], population[:, 1])

        return population

    def Mutation(self):
        """Мутация"""
        delta = (np.random.rand(*self.population_.shape) - 0.5) / MUTATE_CONSTANT
        self.population_ = self.population_ + delta
        x_column = self.population_[:][0]
        y_column = self.population_[:][1]

        satisfying_x = np.where(np.logical_and(x_column > self.bounds_[0], x_column < self.bounds_[1]))
        satisfying_y = np.where(np.logical_and(y_column > self.bounds_[2], y_column < self.bounds_[0]))
        if len(satisfying_x) != len(x_column) or len(satisfying_y) != len(y_column):
            self.population_ = self.population_ - delta

    def CrossoverAndCalcFit(self):
        """Кроссовер и вычисление значений фит функции."""
        curr_population = self.population_[self.population_[:, 2].argsort()].copy()

        x3 = curr_population[3][0]
        y3 = curr_population[3][1]
        x2 = curr_population[2][0]
        y2 = curr_population[2][1]
        x1 = curr_population[1][0]
        y1 = curr_population[1][1]

        new_population = np.array(
            [[x3, y1, self.fit_func_(x3, y1)], [x1, y3, self.fit_func_(x1, y3)],
             [x3, y2, self.fit_func_(x3, y2)], [x2, y3, self.fit_func_(x2, y3)]])

        self.max_result_ = np.max(new_population[:, 2])
        self.mean_result_ = np.mean(new_population[:, 2])

        self.population_ = new_population.copy()
