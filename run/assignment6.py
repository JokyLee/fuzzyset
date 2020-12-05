#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.11.15'


import numpy as np
import matplotlib.pyplot as plt

import fzset


def main():
    x = [0.1, 0.5, 0.9, 0.4, 0.2, 1.0, 0.0]
    y = [0.7, 0.6, 0.6, 1.0, 0.0, 0.1, 0.3]
    rel = fzset.indexOfRelationality(x, y)
    print("rel:", rel)

    x = [0.3, 0.2, 0.7, 0.8, 0.6, 0.5, 1.0]
    y = [0.5, 0.1, 0.2, 1.0, 0.9, 0.8, 0.3]
    rel = fzset.indexOfRelationality(x, y)
    print("rel:", rel)

    import skfuzzy as fuzz

    x = np.linspace(0, 10, 100)
    y = x.copy()
    y[30:70] = 3
    y[70:] = 2 * x[70:] - 11
    # y += (np.random.random(len(x)) - 0.5)
    sampled_idx = np.random.choice(range(100), 66)
    sampled_idx = sampled_idx.tolist()
    sampled_idx.sort()
    sampled_idx = np.array(sampled_idx)
    sampled_x = x[sampled_idx]
    sampled_y = y[sampled_idx]
    sampled_x_dxn = sampled_x.reshape(1, -1)

    # for m in range(1, 10):
    m = 2
    c = 5
    error = []
    for m in range(1, 10):
    # for m in np.linspace(1, 5, 100):
        if m == 1:
            cntr, Ax, u0, d, jm, p, fpc = fuzz.cluster.cmeans(sampled_x.reshape(1, -1), c, 1.00000001, error=0.005, maxiter=1000)
        else:
            cntr, Ax, u0, d, jm, p, fpc = fuzz.cluster.cmeans(sampled_x.reshape(1, -1), c, m, error=0.005, maxiter=1000)

        z = []
        for ax in Ax:
            z.append(np.array([ax, ax * sampled_x]).T)
        F = np.hstack(z)
        a_opt = np.linalg.inv(F.T @ F) @ F.T @ np.array(sampled_y).reshape(-1, 1)
        y_hat = F @ a_opt
        e = sum(abs(y_hat.reshape(-1) - sampled_y)) / len(sampled_y)
        error.append(e)
        print(e)
        plt.plot(sampled_x, y_hat, label="y_hat")
        plt.xlabel("x", fontsize=15)
        plt.ylabel("y", fontsize=15)
        plt.scatter(sampled_x, sampled_y)
        plt.title("m={}".format(m), fontsize=20)
        plt.legend()
        plt.show()
    plt.title("Errors in different values of m", fontsize=20)
    plt.xlabel("m", fontsize=15)
    plt.ylabel("error", fontsize=15)
    plt.plot(range(1, 10), error)
    plt.show()


if __name__ == '__main__':
    main()
