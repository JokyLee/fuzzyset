#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.09.18'


import numpy as np
from sympy import *
init_printing(use_unicode=False, wrap_line=False)

import fzset


def energyFunc(u):
    return u * u


def main():
    # 2
    A = [1.0, 0.6, 0.2, 0.4, 0.5, 0.6, 0.7, 0.9]
    assert len(A) == 8
    print(list(map(energyFunc, A)))
    print(fzset.descriptors.energyMeasure(A, energyFunc))

    points = [
        [0, 0],
        [0.2, 0.8],
        [0.5, 1.0],
        [0.8, 0.8],
        [1.0, 0.0],
    ]
    piecewise_mem = fzset.memberships.piecewise.fromPoints(points)
    print(list(map(piecewise_mem.eval, A)))
    print(fzset.descriptors.entropyMeasure(A, piecewise_mem.eval))

    alpha = 0.41
    print("alpha({}) cut of A {}: ".format(alpha, A))
    print(fzset.descriptors.alphaCut(A, alpha))
    print("core of A:", fzset.descriptors.core(A))
    print("support of A:", fzset.descriptors.support(A))

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
