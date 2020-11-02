import numpy as np
from prettytable import PrettyTable


class GeneticAlgorithm:

    def __init__(self):
        self.fit_func = lambda x, y: np.exp(- x ** 2) * np.exp(- y ** 2) / (1 + x ** 2 + y ** 2)
        self.number_of_iterations = 100
        self.bounds = [-2., 2., -2., 2.]
        self.populations_count = 4
        self.gen_number = 0
        self.population = self.InitPopulation()

        self.max_result = np.max(self.population[:, 2])
        self.mean_result = np.mean(self.population[:, 2])

    def PrintPopulation(self):
        print(f"№ поколения: {self.gen_number}")
        table = PrettyTable()
        table.field_names = ["X", "Y", "FIT"]

        for i in range(self.populations_count):
            table.add_row(list(self.population[i]))

        print(table)

        print(f"Максимальный результат: {self.max_result}")
        print(f"Средний результат: {self.mean_result}")

    def Iterating(self):
        for i in range(self.number_of_iterations):
            self.Iterate()
            self.PrintPopulation()

    def InitPopulation(self):
        population = np.random.rand(self.populations_count, 3)

        population[:, 0] = self.bounds[0] + population[:, 0] * (self.bounds[1] - self.bounds[0])
        population[:, 1] = self.bounds[2] + population[:, 1] * (self.bounds[3] - self.bounds[2])

        population[:, 2] = self.fit_func(population[:, 0], population[:, 1])

        return population

    def Iterate(self):
        curr_population = self.population[self.population[:, 2].argsort()].copy()

        delta = (np.random.rand(*self.population.shape) - 0.5) / 10
        # TODO : проверка выхода за границы диапазона.
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
