from scipy import integrate
import numpy as np
from math import *
from numpy import inf
import matplotlib.pyplot as plt
import scipy.stats as stats
from tabulate import tabulate

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
            distr = array_u
            plt.title(r'Uniform Distribution, N=%i' % size)
            plt.savefig("Uniform, 100" + ".png", format='png')
            plt.show()
        print('Uniform ', size, ': ', np.around(check_normal(array_u), decimals=2))


    #check chi2
    s = 100
    #distr = stats.uniform.rvs(-np.sqrt(3), 2 * np.sqrt(3), size=100)

    alpha = 0.05
    p = 1 - alpha
    k = 6

    limits = np.linspace(-1.4, 1.4, num=k - 1)
    sample = stats.chi2.ppf(p, k - 1)
    array = np.array([stats.norm.cdf(limits[0])])
    quan_ar = np.array([len(distr[distr <= limits[0]])])
    for i in range(0, len(limits) - 1):
        new_ar = stats.norm.cdf(limits[i + 1]) - stats.norm.cdf(limits[i])
        array = np.append(array, new_ar)
        quan_ar = np.append(quan_ar, len(distr[(distr <= limits[i + 1]) & (distr >= limits[i])]))
    array = np.append(array, 1 - stats.norm.cdf(limits[-1]))
    quan_ar = np.append(quan_ar, len(distr[distr >= limits[-1]]))
    result = np.divide(np.multiply((quan_ar - s * array), (quan_ar - s * array)), array * s)


    headers = ["i", "limits", "n_i", "p_i", "np_i", "n_i - np_i", "...^2"]
    rows = []
    for i in range(0, len(quan_ar)):
        if i == 0:
            boarders = ['-inf', np.around(limits[0], decimals=2)]
        elif i == len(quan_ar) - 1:
            boarders = [np.around(limits[-1], decimals=2), 'inf']
        else:
            boarders = [np.around(limits[i - 1], decimals=2), np.around(limits[i], decimals=2)]
        rows.append([i + 1, boarders, quan_ar[i], np.around(array[i], decimals=4), np.around(array[i] * s, decimals=2),
                     np.around(quan_ar[i] - s * array[i], decimals=2), np.around(result[i], decimals=2)])
    rows.append([len(quan_ar), "-", np.sum(quan_ar), np.around(np.sum(array), decimals=4),
                 np.around(np.sum(array * s), decimals=2),
                 -np.around(np.sum(quan_ar - s * array), decimals=2),
                 np.around(np.sum(result), decimals=2)])
    print(tabulate(rows, headers, tablefmt="latex"))

    print(len(quan_ar))
    print('\n')

