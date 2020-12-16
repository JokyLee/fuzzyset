#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.10.13'


import numpy as np
import fzset


def main():
    x = [-6, -5, -3, -1, 0, 2, 3, 5, 6]
    Ax = [1.0, 0.6, 0.7, 1.0, 0.5, 0.4, 0.8, 0.9, 0.9]
    print(fzset.mapTo(x, Ax, abs))
    print(fzset.mapTo(x, Ax, lambda x: x * x))
    print(fzset.mapTo(x, Ax, lambda x: x ** 0.5))

    A = fzset.DiscreteMembership([0.7, 0.2, 0.1, 0.9, 1.0])
    B = fzset.DiscreteMembership([0.4, 0.6, 1.0, 0.5, 0.1])
    print((A.complement().union(B)).intersection(A))

    a = [0.4, 0.7, 0.9, 0.1, 0.6]
    w = [1 / 5.0] * 5
    res = fzset.orderedWeightedAverage(a, w)
    print(res)

    U = np.array([
        [0.3, 1.0, 0.4, 0.0, 0.1],
        [0.0, 0.0, 0.3, 0.9, 0.5],
        [0.7, 0.0, 0.3, 0.1, 0.4],
    ])
    P = fzset.calProximityMatrix(U)
    print(P)


if __name__ == '__main__':
    main()
