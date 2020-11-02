# Copyright 2020 Alexey Alexandrov <sks2311211@yandex.ru>

import genetic_algorithm as ga

if __name__ == '__main__':
    genetic = ga.GeneticAlgorithm()
    print(genetic.population)
    genetic.PrintPopulation()
    genetic.Iterating()
