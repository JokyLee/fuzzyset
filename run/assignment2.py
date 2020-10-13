#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.09.29'


import fzset
from sympy import *


def implication_min(a, b):
    if a <= b:
        return 1
    return b

def implication_algeproduct(a, b):
    if a <= b:
        return 1
    return b / a


def solveR(x, y, implicationFunc):
    R = MutableDenseNDimArray([0] * (len(x) * len(y)), (len(x), len(y)))
    for i, a in enumerate(x):
        for j, b in enumerate(y):
            R[i, j] = implicationFunc(a, b)
    return R


def main():
    # X = Array([1, 0.5, 0.2, 0.7])
    # Y = Array([0.3, 0.6, 0.8])
    # R = solveR(X, Y, implication_min)
    # pprint(R)

    # X1 = Matrix([1, 0, 0, 0])
    # Y1 = Matrix([0.6, 0.2, 0.0, 0.7, 1.0, 0.6])
    # R1 = solveR(X1, Y1, implication_algeproduct)
    # print("R1")
    # pprint(R1)
    #
    # X2 = Matrix([0, 1, 0, 0])
    # Y2 = Matrix([0.2, 0.1, 0.7, 0.9, 1.0, 0.2])
    # R2 = solveR(X2, Y2, implication_algeproduct)
    # print("R2")
    # pprint(X2)
    # pprint(Y2)
    # pprint(R2)
    #
    # X3 = Matrix([0, 0, 0, 1])
    # Y3 = Matrix([0.6, 0.7, 0.5, 0.7, 1.0, 0.0])
    # R3 = solveR(X3, Y3, implication_algeproduct)
    # print("R3")
    # pprint(X3)
    # pprint(Y3)
    # pprint(R3)
    #
    # X4 = Matrix([0, 0, 1, 0])
    # Y4 = Matrix([0.0, 1.0, 0.0, 0.7, 0.2, 0.4])
    # R4 = solveR(X4, Y4, implication_algeproduct)
    # print("R4")
    # pprint(X4)
    # pprint(Y4)
    # pprint(R4)

    w0 = 3
    w1 = fzset.Interval([-3, 2])
    w2 = fzset.Interval([1, 4])
    x1 = 0.5
    x2 = -4
    a = w1 * x1
    print(a)
    b = w2 * x2
    print(b)
    u = a + b + w0
    print(u)

    w0 = 3
    w1 = fzset.Interval([-3, 2])
    w2 = fzset.Interval([1, 4])
    x1 = fzset.Interval([0.2, 0.7])
    x2 = fzset.Interval([-5.0, -1.0])
    a = w1 * x1
    print(a)
    b = w2 * x2
    print(b)
    u = a + b + w0
    print(u)



if __name__ == '__main__':
    main()
