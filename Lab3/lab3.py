import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import scipy.special as sc

distr_type = ['Normal', 'Cauchy', 'Laplace', 'Poisson', 'Uniform']


def get_distr_samples(d_name, num):
    if d_name == 'Normal':
        return np.random.normal(0, 1, num)
    elif d_name == 'Cauchy':
        return np.random.standard_cauchy(num)
    elif d_name == 'Laplace':
        return np.random.laplace(0, np.sqrt(2) / 2, num)
    elif d_name == 'Poisson':
        return np.random.poisson(10, num)
    elif d_name == 'Uniform':
        return np.random.uniform(-np.sqrt(3), np.sqrt(3), num)
    return []


def get_distr_func(d_name, x):
    if d_name == 'Normal':
        return 0.5 * (1 + sc.erf(x / np.sqrt(2)))
    elif d_name == 'Cauchy':
        return np.arctan(x) / np.pi + 0.5
    elif d_name == 'Laplace':
        if x <= 0:
            return 0.5 * np.exp(np.sqrt(2) * x)
        else:
            return 1 - 0.5 * np.exp(-np.sqrt(2) * x)
    elif d_name == 'Poisson':
        return sc.gammainc(x + 1, 10) / sc.factorial(x)
    elif d_name == 'Uniform':
        if x < -np.sqrt(3):
            return 0
        elif np.fabs(x) <= np.sqrt(3):
            return (x - np.sqrt(3)) / 2 * np.sqrt(3)
        else:
            return 1
    return 0


quantity = [20, 100]
repeat = 1000

if __name__ == '__main__':
    headers = ["distribution name", "proportion of ejections", "dispersion"]
    headers_th = ["distibution name", "q_1", "q_3", "x_1", "x_2", "p"]
    rows = []
    rows_th = []
    prob = []
    disp = []
    for dist_name in distr_type:
        array_20 = get_distr_samples(dist_name, quantity[0])
        array_100 = get_distr_samples(dist_name, quantity[1])
        line_props = dict(color="black", alpha=0.3, linestyle="dashdot")
        bbox_props = dict(color="b", alpha=0.9)
        flier_props = dict(marker="o", markersize=4)
        plt.boxplot((array_20, array_100), whiskerprops=line_props, boxprops=bbox_props, flierprops=flier_props, labels=["n = 20", "n = 100"])
        plt.ylabel("X")
        plt.title(dist_name)
        plt.savefig(dist_name + '.png', format='png')
        plt.show()

        array = []
        for size in quantity:
            out = np.zeros(repeat)
            for i in range(0, repeat):
                array = get_distr_samples(dist_name, size)
                x = []
                x.append(np.quantile(array, 0.25) - 1.5 * (np.quantile(array, 0.75) - np.quantile(array, 0.25)))
                x.append(np.quantile(array, 0.75) + 1.5 * (np.quantile(array, 0.75) - np.quantile(array, 0.25)))

                for k in range(0, size):
                    if array[k] > x[1] or array[k] < x[0]:
                        out[i] += 1

            out /= size
            prob = sum(out) / repeat
            disp = (1 / repeat) * sum(out * out) - prob * prob

            rows.append([dist_name + ", n = " + str(size),  np.around(prob, decimals=3), np.around(disp, decimals=5)])

    print(tabulate(rows, headers, tablefmt="latex"))
    print("\n")

