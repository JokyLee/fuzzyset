#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.09.18'


import numpy as np
from sympy import *
init_printing(use_unicode=False, wrap_line=False)

import fzset


def solvePieceWiseLinear(points):
    coeffs = []
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        # ax + b = y
        ab = np.linalg.solve([[x0, 1], [x1, 1]], [y0, y1])
        coeffs.append(ab)
    return coeffs


def entropyMeasure(hFunc, points):
    res = []
    x = Symbol('x')
    for i, (a, b) in enumerate(hFunc):
        res.append(integrate(a * x + b, (x, points[i][0], points[i + 1][0])))
    return sum(res), res


def energyFunc(u):
    return u * u


def energyMeasure(A, energyFunction):
    return sum(map(energyFunction, A))


def alphaCut(A, alpha):
    cut = []
    for a in A:
        if a > alpha:
            cut.append(1)
        else:
            cut.append(0)
    return cut


def getCore(A):
    return list(map(lambda i: 1 if i == 1 else 0, A))


def getSupport(A):
    return list(map(lambda i: 1 if i > 0 else 0, A))


def main():
    A = [1.0, 0.6, 0.2, 0.4, 0.5, 0.6, 0.7, 0.9]
    assert len(A) == 8
    print(energyMeasure(A, energyFunc))

    points = [
        [0, 0],
        [0.2, 0.8],
        [0.5, 1.0],
        [0.8, 0.8],
        [1.0, 1.0],
    ]
    coeffs = solvePieceWiseLinear(points)
    for i, (a, b) in enumerate(coeffs):
        print("y = {} * x + {}, {} <= x <= {}".format(a, b, points[i][0], points[i + 1][0]))

    print("---------------- integral start -----------------")
    for (a, b) in coeffs:
        x = Symbol('x')
        print(integrate(a * x + b, x))
    print("---------------- integral end -----------------")

    res, integrals = entropyMeasure(coeffs, points)
    print("{} = {}".format(res, integrals))

    alpha = 0.41
    print("alpha({}) cut of A {}: ".format(alpha, A))
    print(alphaCut(A, alpha))
    print("core of A:", getCore(A))
    print("support of A:", getSupport(A))

    x = Symbol('x')
    print(integrate(1 - asin(x) / 5, x))
    res = integrate(1 - asin(x) / 5, (x, 0, 1))
    print(res)

    print("--------------------------------------")
    A = fzset.Interval([1, 5])
    B = fzset.Interval([-2, 4])
    C = fzset.Interval([1, 2])
    print(A + B)
    print(A * 2)
    print(C * B)
    print(A * 2 + C * B)

    print("--------------------------------------")
    a = fzset.Interval((-1, 1))
    b = fzset.Interval((-2, 1))
    print(a * A)
    print(b * B)
    print(a * A + b * B)
    print((a * A + b * B) / C)


if __name__ == '__main__':
    main()
