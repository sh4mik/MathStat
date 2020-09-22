from scipy import integrate
import numpy as np
from math import *
from numpy import inf
import matplotlib.pyplot as plt


def check_normal(data):
    m = np.sum(data) / len(data)
    sigma = np.std(data)
    n = len(data)
    coef = 1 / (sigma * np.sqrt(np.pi * 2))
    function_g = lambda t: coef * np.exp(-(t - m)**2 / (2*sigma))
    sum = 0
    i = 1
    data = sorted(data)
    for x in data:
        integr = integrate.quad(function_g, -inf, x)
        sum = sum + (integr[0] - (2 * i - 1) / (2 * n))**2
        i = i + 1
    sum += 1 / (12 * n)
    return sum


bins = 20
num = [10, 100, 500]
if __name__ == '__main__':
    for size in num:
        array = np.random.normal(0, 1, size)
        sum = check_normal(array)
        print(np.around(sum, decimals=2))

    #check flexibility
    for size in num:
        array_u = np.random.uniform(-np.sqrt(3), np.sqrt(3), size)
        if size == 100:
            n, bins, patches = plt.hist(array_u, bins, density=1, facecolor='grey', edgecolor='black', alpha=0.2)
            plt.title(r'Uniform Distribution, N=%i' % size)
            plt.savefig("Uniform, 100" + ".png", format='png')
            plt.show()
        print('Uniform ', size, ': ', np.around(check_normal(array_u), decimals=2))