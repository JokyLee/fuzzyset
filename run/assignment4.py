#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.10.29'


import math

import numpy as np
import matplotlib.pyplot as plt

import fzset


def createParabolic(m, a):
    lower_bound = m - a
    upper_bound = m + a
    x = np.linspace(lower_bound, upper_bound, 100)
    y = 1 - ((x - m) / a) ** 2
    return x, y


def plotParabolic():
    a1 = 2
    m1 = 2
    a2 = 2.5
    m2 = 5
    a3 = 2
    m3 = 8
    points1 = createParabolic(m1, a1)
    points2 = createParabolic(m2, a2)
    points3 = createParabolic(m3, a3)

    fig, ax = plt.subplots()  # Create a figure and an axes.
    ax.plot(*points1, label='1')  # Plot some data on the axes.
    ax.plot(*points2, label='2')  # Plot some data on the axes.
    ax.plot(*points3, label='3')  # Plot some data on the axes.
    # ax.plot(x, x ** 2, label='quadratic')  # Plot more data on the axes...
    # ax.plot(x, x ** 3, label='cubic')  # ... and some more.
    # ax.set_xlabel('x label')  # Add an x-label to the axes.
    # ax.set_ylabel('y label')  # Add a y-label to the axes.
    # ax.set_title("Simple Plot")  # Add a title to the axes.
    ax.legend()  # Add a legend.
    plt.ylim([0, 1.05])
    plt.show()


def ahp():
    # grades = [8, 4.5, 3, 2, 5, 9, 7.5, 1]
    # for i, g in enumerate(grades):
    #     print("{}: ".format(i), end='')
    #     for j in grades[i:]:
    #         print("{:.2f} ".format(j / g), end='')
    #     print()
    R = np.array([
        [1, 2, 3, 4, 2, 1/3, 1, 9],
        [1/2, 1, 2, 2, 1/2, 1/5, 1/4, 5],
        [1/3, 1/2, 1, 2, 1/2, 1/5, 1/4, 3],
        [1/4, 1/2, 1/2, 1, 1/3, 1/5, 1/4, 3],
        [1/2, 2, 2, 3, 1, 1/3, 1/2, 5],
        [3, 5, 5, 5, 3, 1, 2, 9],
        [1, 4, 4, 4, 2, 1/2, 1, 8],
        [1/9, 1/5, 1/3, 1/3, 1/5, 1/9, 1/8, 1]
    ])
    inconsistency_index, l_max, membership = fzset.analyticHierarchyProcess(R)
    print(inconsistency_index, l_max, membership)


def paramsOfFuzzyEvent():
    x = [-6, -5, -3, -1, 0, 2, 3, 5, 6]
    Ax = [1.0, 0.6, 0.7, 1.0, 0.5, 0.4, 0.8, 0.9, 0.9]
    px = [0.15, 0.05, 0.1, 0.2, 0, 0.4, 0.05, 0.05, 0.0]
    probability = fzset.calProbabilityOfFuzzyEvent(x, Ax, px)
    print("probability:", probability)
    mean = fzset.calMeanOfFuzzyEvent(x, Ax, px)
    print("mean:", mean)
    variance = fzset.calVarianceOfFuzzyEvent(x, Ax, px)
    print("variance:", variance)


def justifiableGranularity():
    experimental_data = np.array([
        0, 1.3, 5.1, 4.6, 7.5, 2.2, 1.7, 3.3, 4.1, 4.2, 5.0, 8.1, 9.0
    ])
    r = experimental_data.mean()
    print("r = {}".format(r))


def justifiableGranularity2():
    experimental_data = np.array([
        8.7, 8.5, 4.3, 2.9, 0.8, 1.5, 1.6, 2.4, 1.1, 5.1, 4.8, 4.5, 6.1, 3.1, 7.9, 8.1, 2.0
    ])
    experimental_data.sort()
    r = experimental_data.mean()
    print("r = {} round to {}".format(r, round(r, 2)))
    r = round(r, 2)
    print("================= upper bound =================")
    upper_range = experimental_data.max() - r
    print("experimental_data.max(): {}, upper_range = {}".format(experimental_data.max(), upper_range))
    upper = experimental_data[experimental_data > r]
    print(upper, len(upper))
    for i, b in enumerate(upper):
        cov = (i + 1) / len(upper)
        sp = 1 - abs(b - r) / upper_range
        v = cov * sp
        print("{}: cov {}, sp {}, v {}".format(b, cov, sp, v))

    print("================= lower bound =================")
    lower_range = r - experimental_data.min()
    print("experimental_data.min(): {}, upper_range = {}".format(experimental_data.min(), lower_range))
    lower = experimental_data[experimental_data < r]
    print(lower, len(lower))
    for i, a in enumerate(lower[::-1]):
        cov = (i + 1) / len(lower)
        sp = 1 - abs(a - r) / lower_range
        v = cov * sp
        print("{}: cov {}, sp {}, v {}".format(a, cov, sp, v))


def justifiableGranularity3():
    experimental_data = np.array([
        0, 1.3, 5.1, 4.6, 7.5, 2.2, 1.7, 3.3, 4.1, 4.2, 5.0, 8.1, 9.0
    ])
    space = [-2, 12]
    Ax = lambda x: max(0, math.cos(x))
    experimental_data.sort()
    r = experimental_data.mean()
    print("r = {} round to {}".format(r, round(r, 2)))
    r = round(r, 2)
    print("================= upper bound =================")
    upper_range = experimental_data.max() - r
    print("experimental_data.max(): {}, upper_range = {}".format(experimental_data.max(), upper_range))
    upper = experimental_data[experimental_data > r]
    print(upper, len(upper))
    for i, b in enumerate(upper):
        covs = []
        for j in range(i + 1):
            covs.append(Ax(upper[j]))
        cov = sum(covs) / len(upper)
        sp = 1 - abs(b - r) / upper_range
        v = cov * sp
        print("{}: cov {}, sp {}, v {}".format(b, cov, sp, v))

    print("================= lower bound =================")
    lower_range = r - experimental_data.min()
    print("experimental_data.min(): {}, upper_range = {}".format(experimental_data.min(), lower_range))
    lower = experimental_data[experimental_data < r]
    print(lower, len(lower))
    for i, a in enumerate(lower[::-1]):
        covs = []
        for j in range(i + 1):
            covs.append(Ax(lower[j]))
        cov = sum(covs) / len(lower)
        sp = 1 - abs(a - r) / lower_range
        v = cov * sp
        print("{}: cov {}, sp {}, v {}".format(a, cov, sp, v))


def Q5():
    cov = lambda a: 1/a * (np.sin(7.68 * a) - np.sin(-6.32 * a))
    sp = lambda a: 0.78 - 1 / (7 * a)
    for a in [0.4, 0.35, 0.3, 0.25, 0.2]:
        V = cov(a) * sp(a)
        print(a, V)


def main():
    # paramsOfFuzzyEvent()
    # plotParabolic()
    # ahp()
    # justifiableGranularity()
    # justifiableGranularity2()
    # justifiableGranularity3()
    Q5()


if __name__ == '__main__':
    main()
