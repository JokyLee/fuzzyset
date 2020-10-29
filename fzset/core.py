#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.09.18'
__all__ = [
    "Interval",
    "DiscreteMembership",
    "supMinComposition",
    "infMaxComposition",
    "calProximityMatrix",
]


import numpy as np

import itertools


def supMinComposition(R, G):
    assert R.shape[1] == G.shape[0]
    T = np.zeros((R.shape[0], G.shape[1]))
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            T[i, j] = max(map(min, zip(R[i, :], G[:, j])))
    return T


def infMaxComposition(R, G):
    assert R.shape[1] == G.shape[0]
    T = np.zeros((R.shape[0], G.shape[1]))
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            T[i, j] = min(map(max, zip(R[i, :], G[:, j])))
    return T


class Interval:
    def __init__(self, interval):
        assert isinstance(interval, (list, tuple))
        assert len(interval) == 2
        assert interval[1] >= interval[0]
        self._interval = tuple(interval)

    @property
    def min(self):
        return self._interval[0]

    @property
    def max(self):
        return self._interval[1]

    def __str__(self):
        return self._interval.__str__()

    def __eq__(self, other):
        assert isinstance(other, Interval)
        return self.min == other.min and self.max == other.max

    def __ne__(self, other):
        return not self.__eq__(other)

    def __add__(self, other):
        if isinstance(other, Interval):
            lower = self.min + other.min
            upper = self.max + other.max
            return Interval((lower, upper))
        elif isinstance(other, (int, float)):
            lower = self.min + other
            upper = self.max + other
            return Interval((lower, upper))
        else:
            raise ValueError("must be (Interval, int, float)")

    def __mul__(self, other):
        if isinstance(other, Interval):
            candidates = [a * b for (a, b) in itertools.product(self._interval, other._interval)]
            return Interval((min(candidates), max(candidates)))
        elif isinstance(other, (int, float)):
            lower = self.min * other
            upper = self.max * other
            if other < 0:
                return Interval((upper, lower))
            return Interval((lower, upper))
        else:
            raise ValueError("must be (Interval, int, float)")

    def __sub__(self, other):
        assert isinstance(other, Interval)
        return self + Interval((-1, -1)) * other

    def __truediv__(self, other):
        assert isinstance(other, Interval)
        return self * Interval((1 / other.max, 1 / other.min))

    def __len__(self):
        return self.max - self.min

    def intersection(self, interval):
        assert isinstance(interval, Interval)
        lower = max(self.min, interval.min)
        upper = min(self.max, interval.max)
        return Interval((lower, upper))

    def union(self, interval):
        assert isinstance(interval, Interval)
        lower = min(self.min, interval.min)
        upper = max(self.max, interval.max)
        return Interval((lower, upper))


class DiscreteMembership:
    def __init__(self, values):
        self._values = values

    def __str__(self):
        return self._values.__str__()

    def complement(self):
        new_values = [1 - v for v in self._values]
        return DiscreteMembership(new_values)

    def union(self, other):
        assert isinstance(other, DiscreteMembership)
        new_values = [max(a, b) for a, b in zip(self._values, other._values)]
        return DiscreteMembership(new_values)

    def intersection(self, other):
        assert isinstance(other, DiscreteMembership)
        new_values = [min(a, b) for a, b in zip(self._values, other._values)]
        return DiscreteMembership(new_values)


def calProximityMatrix(partitionMatrix):
    n = partitionMatrix.shape[1]
    proximity = np.zeros((n, n))
    for k in range(n):
        for l in range(n):
            proximity[k, l] = sum(np.min(partitionMatrix[:, (k, l)], axis=1))
    return proximity
