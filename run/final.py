#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.12.17'

import numpy as np
import skfuzzy as fuzz
from sympy import *
import matplotlib.pyplot as plt

import fzset


init_printing(use_unicode=False, wrap_line=False)


def main():
    data = np.loadtxt('../data/data_banknote_authentication.txt', delimiter=',')
    print(data.shape)
    print(data[:10])
    np.random.shuffle(data)
    print(data.shape)
    print(data[:10])

    # split train and test
    sample_size = data.shape[0]
    split_loc = int(0.7 * sample_size)
    training_set = data[:split_loc]
    training_x = training_set[:, :4]
    training_y = training_set[:, -1].astype(np.int)
    testing_set = data[split_loc:]
    testing_x = testing_set[:, :4]
    testing_y = testing_set[:, -1].astype(np.int)
    plot_data = []
    # c = 10
    for c in range(2, 100):
    # for c in range(100, 500, 30):
        print("c =", c)
        m = 2
        cntr, Ax, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            training_x.T, c, m, error=0.005, maxiter=1000
        )
        cluster_labels = np.argmax(Ax, axis=0)
        cluster_map2_class = {}
        correct_count = 0
        for i in range(c):
            pickup = training_y[cluster_labels == i]
            class1_count = pickup.sum()
            if class1_count >= len(pickup) / 2:
                cluster_map2_class[i] = 1
                correct_count += class1_count
            else:
                cluster_map2_class[i] = 0
                correct_count += len(pickup) - class1_count

        training_error = 1 - correct_count / len(training_y)
        print(training_error)

        Ax_testing, _, _, _, _, _ = fuzz.cmeans_predict(testing_x.T, cntr, m, error=0.005, maxiter=1000)
        # Ax_testing, _, _, _, _, _ = fuzz.cmeans_predict(training_x.T, cntr, m, error=0.005, maxiter=1000)
        cluster_labels_testing = np.argmax(Ax_testing, axis=0)
        correct_testing = 0
        for l, gt in zip(cluster_labels_testing, testing_y):
        # for l, gt in zip(cluster_labels_testing, training_y):
            if cluster_map2_class[l] == gt:
                correct_testing += 1
        testing_error = 1 - correct_testing / len(testing_y)
        # testing_error = 1 - correct_testing / len(training_y)
        print(testing_error)
        classification_error = 1 - (correct_count + correct_testing) / (len(training_y) + len(testing_y))
        plot_data.append([c, training_error, testing_error, classification_error])
    np.savetxt("../data/results_all.txt", np.array(plot_data))


def plotError():
    import matplotlib.pyplot as plt
    # plot_data = np.loadtxt("../data/results.txt")
    plot_data = np.loadtxt("../data/results_all.txt")
    # plot_data = np.loadtxt("../data/results_combine.txt")
    x = plot_data[:, 0]
    training_error = plot_data[:, 1]
    testing_error = plot_data[:, 2]
    classification_error = plot_data[:, 3]
    plt.plot(x, training_error, label="training error")
    plt.plot(x, testing_error, label="testing error")
    plt.plot(x, classification_error, label="classification error")
    plt.xlabel('number of clusters')
    plt.ylabel('error')
    plt.legend()
    plt.show()


def Q6():
    x = Symbol('x')
    r = Symbol('r')
    b = Symbol('b')
    pprint(integrate(1 - 2*b*sqrt(1 - x)/r, x))
    res = integrate(1 - 2*b*sqrt(1 - x)/r, (x, 0, 1))
    print(res)


def Q7():
    x = np.linspace(0, 11, 1100)
    A1 = fzset.memberships.triangular(0.5, 2, 3.5)
    A2 = fzset.memberships.triangular(2, 3.5, 7)
    A3 = fzset.memberships.triangular(3.5, 7, 9)
    A4 = fzset.memberships.triangular(7, 9, 11)
    As = [A1, A2, A3, A4]
    Ay = np.array([[A.eval(i) for i in x] for A in As])
    y_out = np.array([5.0, 4.0, -2.0, 6.0])
    ys = (Ay * y_out.reshape(-1, 1)).sum(axis=0)
    # ys = []
    # for idx in np.argmax(Ay, axis=0):
    #     ys.append(y_out[idx])

    plt.subplot(2, 1, 1)
    plt.plot(x[50:-1], ys[50:-1], label="y_hat")
    plt.ylabel("y", fontsize=15)

    plt.subplot(2, 1, 2)
    for i, Ax in enumerate(Ay):
        plt.plot(x, Ax, label="A{}".format(i + 1))
    plt.xlim(0, 11)
    # plt.ylim(0, 1)
    plt.xlabel("x", fontsize=15)
    plt.ylabel("membership", fontsize=15)
    plt.xticks(range(0, 12))
    plt.legend()
    plt.show()

    x = 7.5
    res = (np.array([A.eval(x) for A in As]) * np.array([5.0, 4.0, -2.0, 6.0])).sum()
    print(res)

    x = 3.7
    res = (np.array([A.eval(x) for A in As]) * np.array([5.0, 4.0, -2.0, 6.0])).sum()
    print(res)


def Q9():
    x = Symbol('x')
    s = Symbol('s')
    px = (1 / (s * sqrt(2 * pi))) * exp(-(x*x) / (2*s*s))
    pprint(px)
    pprint(integrate((x - 1) / 2 * px, x))
    res = integrate(integrate((x - 1) / 2 * px, x), (x, 1, 1.5))
    print(res)

    pprint(integrate((2 - x) / 2 * px, x))
    res = integrate(integrate((2 - x) / 2 * px, x), (x, 1.5, 2.0))
    print(res)
    # print(integrate(1 - (x/5)**2, (x, -5, 5)))
    # print(integrate((x+10)/10, (x, -10, 0)))


if __name__ == '__main__':
    # main()
    plotError()
    # Q6()
    # Q7()
    # Q9()
