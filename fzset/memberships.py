#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.12.14'
__all__ = [
    "piecewise",
]

import bisect

import numpy as np


def solvePieceWiseLinear(points):
    coeffs = []
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        # ax + b = y
        ab = np.linalg.solve([[x0, 1], [x1, 1]], [y0, y1])
        coeffs.append(ab)
    return coeffs

class piecewise:
    def __init__(self, xIntervals, coeffs):
        assert len(xIntervals) == len(coeffs) + 1
        assert xIntervals[0] == 0 and xIntervals[-1] == 1
        for a, b in zip(xIntervals[:-1], xIntervals[1:]):
            assert a < b
        self._x_interval = xIntervals
        self._coeffs = coeffs

    def printFuncs(self):
        for i, (a, b) in enumerate(self._coeffs):
            print("y = {} * x + {}, {} <= x <= {}".format(
                a, b, self._x_interval[i][0], self._x_interval[i + 1][0])
            )

    @classmethod
    def fromPoints(cls, points_nx2):
        coeffs = solvePieceWiseLinear(points_nx2)
        return cls([x for (x, _) in points_nx2], coeffs)

    def eval(self, x):
        idx = bisect.bisect_right(self._x_interval, x) - 1
        if idx >= len(self._coeffs):
            idx = len(self._coeffs) - 1
        return self._coeffs[idx][0] * x + self._coeffs[idx][1]
