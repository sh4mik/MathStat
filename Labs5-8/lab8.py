import numpy as np
import scipy.stats as stats

gamma = 0.95


def m_confidence_interval(d):
    m = np.mean(d)
    s = np.std(d)
    n = len(d)
    interval = s * stats.t.ppf((1 + gamma) / 2, n - 1) / (n - 1) ** 0.5
    print('mean: ', np.around(m - interval, decimals=2), ' ', np.around(m + interval, decimals=2))


def var_confidence_interval(d):
    s = np.std(d)
    n = len(d)
    low_b = s * (n / stats.chi2.ppf((1 + gamma) / 2, n - 1)) ** 0.5
    high_b = s * (n / stats.chi2.ppf((1 - gamma) / 2, n - 1)) ** 0.5
    print('variance: ', np.around(low_b, decimals=2), ' ', np.around(high_b, decimals=2))


def m_confidence_asymptotic(d):
    m = np.mean(d)
    s = np.std(d)
    n = len(d)
    u = stats.norm.ppf((1 + gamma) / 2)
    interval = s * u / (n ** 0.5)
    print('asymptotic mean: ', np.around(m - interval, decimals=2), ' ', np.around(m + interval, decimals=2))


def var_confidence_asymptotic(d):
    s = np.std(d)
    n = len(d)
    m_4 = stats.moment(d, 4)
    ep = m_4 / s**4 - 3
    u = stats.norm.ppf((1 + gamma) / 2)
    U = u * (((ep + 2) / n) ** 0.5)
    low_b = s * (1 + 0.5 * U) ** (-0.5)
    high_b = s * (1 - 0.5 * U) ** (-0.5)
    print ('asymptotic variance: ', np.around(low_b, decimals=2), ' ', np.around(high_b, decimals=2))


if __name__ == '__main__':
    size = [20, 100]
    for s in size:
        d = np.random.normal(0, 1, size=s)
        print('size = ' + str(s))
        m_confidence_interval(d)
        var_confidence_interval(d)
        m_confidence_asymptotic(d)
        var_confidence_asymptotic(d)
