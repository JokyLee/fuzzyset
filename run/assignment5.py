#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.11.15'


import numpy as np
import matplotlib.pyplot as plt

import fzset


def main():
    # x = [-3, -2.5, -1.0, 0.0, 0.6, 1.7, 3.6, 5.0, 5.5]
    # Ax = [0.3, 0.5, 0.7, 1.0, 0.9, 0.5, 0.3, 0.1, 0.1]
    # gammas = [0, 1.0, 0.5]
    # for g in gammas:
    #     print(fzset.decodeByCenterOfGravity(x, Ax, g))

    # Exp 5
    # a = np.array([-3, 1, 1.5]) + np.array([0.3, 0.6, 0.9]) * 1.5
    # b = np.array([0.3, 0.7, 1.4]) * 2.0
    # c = np.array([-1.1, -0.6, 0.3]) * -3.1
    # print(a, b, c, a + b + c)

    # Exp 6
    # a = fzset.Interval([3, 4])
    # b = fzset.Interval([2, 5])
    # X = fzset.Interval([-3, 2])
    # Y = a * X + b
    # print(Y)

    x = np.linspace(-5, 5, 100)
    y = 3 * x + 0.3 * x * x - 0.1 * x * x * x + (np.random.random(len(x)) - 0.5) * 3

    plt.subplot(2, 1, 1)

    z = []
    sampled_idx = np.random.choice(range(100), 66)
    sampled_x = x[sampled_idx]
    sampled_y = y[sampled_idx]
    all_Ax = []
    for m in np.linspace(-5, 5, 5):
        Ax = np.exp(-(x - m) ** 2)
        all_Ax.append(Ax)

    sampled_Ax = [Ax[sampled_idx] for Ax in all_Ax]
    for Ax in sampled_Ax:
        z.append(np.array([Ax, Ax * np.cos(sampled_x)]).T)
    F = np.hstack(z)
    a_opt = np.linalg.inv(F.T @ F) @ F.T @ np.array(sampled_y).reshape(-1, 1)
    print(a_opt)

    all_z = []
    for Ax in all_Ax:
        all_z.append(np.array([Ax, Ax * np.cos(x)]).T)
    all_F = np.hstack(all_z)
    y_hat = all_F @ a_opt
    # print(y_hat)
    plt.plot(x, y_hat, label="y_hat")
    plt.scatter(sampled_x, sampled_y)
    plt.xlim(-5, 5)
    plt.ylabel("y", fontsize=15)
    plt.xticks(range(-5, 6))
    plt.legend()

    plt.subplot(2, 1, 2)
    for i, Ax in enumerate(all_Ax):
        plt.plot(x, Ax, label="A{}".format(i + 1))
    plt.xlim(-5, 5)
    # plt.ylim(0, 1)
    plt.xlabel("x", fontsize=15)
    plt.ylabel("membership", fontsize=15)
    plt.xticks(range(-5, 6))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
