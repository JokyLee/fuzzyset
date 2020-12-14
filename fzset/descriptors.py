#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.12.14'
__all__ = [
    "core",
    "support",
    "alphaCut",
    "energyMeasure",
    "entropyMeasure",
]


import numpy as np


def core(A):
    return list(map(lambda i: 1 if i == 1 else 0, A))


def support(A):
    return list(map(lambda i: 1 if i > 0 else 0, A))


def alphaCut(A, alpha):
    cut = []
    for a in A:
        if a > alpha:
            cut.append(1)
        else:
            cut.append(0)
    return cut


def energyMeasure(A, energyFunction):
    assert np.allclose([energyFunction(0), energyFunction(1)], [0, 1])
    return sum(map(energyFunction, A))


def entropyMeasure(A, entropyFunction):
    assert np.allclose(
        [entropyFunction(0), entropyFunction(0.5), entropyFunction(1)],
        [0, 1, 0]
    )
    return sum(map(entropyFunction, A))
