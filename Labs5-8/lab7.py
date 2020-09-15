import numpy as np
from tabulate import tabulate
import scipy.stats as stats

alpha = 0.05
p = 1 - alpha
k = 6
size = 100

if __name__ == '__main__':
    d = np.random.normal(0, 1, size=size)
    print('mu = ', np.around(np.mean(d), decimals=2), ' ', 'sigma = ', np.around(np.std(d), decimals=2))

    limits = np.linspace(-1.3, 1.3, num=k-1)
    sample = stats.chi2.ppf(p, k-1)
    array = np.array([stats.norm.cdf(limits[0])])
    quantity = np.array([len(d[d <= limits[0]])])
    for i in range(0, len(limits) - 1):
        new_ar = stats.norm.cdf(limits[i + 1]) - stats.norm.cdf(limits[i])
        array = np.append(array, new_ar)
        quantity = np.append(quantity, len(d[(d <= limits[i + 1]) & (d >= limits[i])]))
    array = np.append(array, 1 - stats.norm.cdf(limits[-1]))
    quantity = np.append(quantity, len(d[d >= limits[-1]]))
    result = np.divide(np.multiply((quantity - size * array), (quantity - size * array)), array * size)

    headers = ["i", "limits", "n_i", "p_i", "np_i", "n_i - np_i", "...2"]
    rows = []
    for i in range(0, len(quantity)):
        if i == 0:
            borders = ['-inf', np.around(limits[0], decimals=2)]
        elif i == len(quantity) - 1:
            borders = [np.around(limits[-1], decimals=2), 'inf']
        else:
            borders = [np.around(limits[i - 1], decimals=2), np.around(limits[i], decimals=2)]
        rows.append([i + 1, borders, quantity[i], np.around(array[i], decimals=4), np.around(array[i] * size, decimals = 2),
                     np.around(quantity[i] - size * array[i], decimals=2), np.around(result[i], decimals=2)])
    rows.append([len(quantity), "-", np.sum(quantity), np.around(np.sum(array), decimals=4),
                 np.around(np.sum(array * size), decimals=2),
                 -np.around(np.sum(quantity - size * array), decimals=2),
                 np.around(np.sum(result), decimals=2)])
    print(tabulate(rows, headers, tablefmt="latex"))
    print('\n')