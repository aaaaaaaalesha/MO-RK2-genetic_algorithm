# Copyright 2020 Alexey Alexandrov <sks2311211@yandex.ru>

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt


def GraphFunc(fit_function):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    xval = np.linspace(-2, 2, 100)
    yval = np.linspace(-2, 2, 100)
    x, y = np.meshgrid(xval, yval)
    z = fit_function(x, y)
    surf = ax.plot_surface(
        x, y, z,
        rstride=2,
        cstride=2,
        cmap=cm.plasma)
    plt.title("График функции f(x, y) = sin(x) / (1 + x^2 + y^2)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
