#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.10.13'


import numpy as np
import fzset


def main():
    A = fzset.DiscreteMembership([0.7, 0.2, 0.1, 0.9, 1.0])
    B = fzset.DiscreteMembership([0.4, 0.6, 1.0, 0.5, 0.1])
    print((A.complement().union(B)).intersection(A))

    a = [0.4, 0.7, 0.9, 0.1, 0.6]
    w = [1 / 5.0] * 5
    print(sum([i * j for i, j in zip(a, w)]))

    U = np.array([
        [0.3, 1.0, 0.4, 0.0, 0.1],
        [0.0, 0.0, 0.3, 0.9, 0.5],
        [0.7, 0.0, 0.3, 0.1, 0.4],
    ])
    P = fzset.calProximityMatrix(U)
    print(P)


if __name__ == '__main__':
    main()
