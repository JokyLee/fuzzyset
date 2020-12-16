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
    "calProbabilityOfFuzzyEvent",
    "calMeanOfFuzzyEvent",
    "calVarianceOfFuzzyEvent",
    "analyticHierarchyProcess",
    "decodeByCenterOfGravity",
    "indexOfRelationality",
    "mapTo",
    "orderedWeightedAverage",
]


import numpy as np
import scipy.linalg

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


def _transFuzzyEventParam(x, Ax, px):
    x = np.array(x, np.float)
    Ax = np.array(Ax, np.float)
    px = np.array(px, np.float)
    assert np.allclose(px.sum(), 1)
    assert len(x) == len(Ax) == len(px)
    return x, Ax, px


def calProbabilityOfFuzzyEvent(x, Ax, px):
    x, Ax, px = _transFuzzyEventParam(x, Ax, px)
    return (Ax * px).sum()


def calMeanOfFuzzyEvent(x, Ax, px):
    x, Ax, px = _transFuzzyEventParam(x, Ax, px)
    return (x * Ax * px).sum()


def calVarianceOfFuzzyEvent(x, Ax, px):
    x, Ax, px = _transFuzzyEventParam(x, Ax, px)
    mean = calMeanOfFuzzyEvent(x, Ax, px)
    return ((x - mean) ** 2 * Ax * px).sum()


def _checkReciprocalMatrix(R):
    for i in range(R.shape[0]):
        for j in range(i, R.shape[1]):
            assert np.allclose(R[i][j], 1/R[j][i])


def analyticHierarchyProcess(R):
    _checkReciprocalMatrix(R)
    eigen_values, eigen_vectors = scipy.linalg.eig(R)
    assert np.allclose(eigen_values[0].imag, 0)
    assert np.allclose(eigen_vectors[:, 0].imag, 0)
    abs_vec = np.abs(eigen_vectors[:, 0].real)
    l_max = eigen_values[0].real
    inconsistency_index = (l_max - len(abs_vec))/ (len(abs_vec) - 1)
    return inconsistency_index, l_max, abs_vec / np.max(abs_vec)


def _transXAndY(x, y):
    x = np.array(x, np.float)
    y = np.array(y, np.float)
    return x, y


def _transXAndAx(x, Ax):
    x, Ax = _transXAndY(x, Ax)
    assert Ax.max() <= 1.0
    return x, Ax


def decodeByCenterOfGravity(x, Ax, gamma):
    x, Ax = _transXAndAx(x, Ax)
    filtered_Ax = Ax[Ax >= gamma]
    filtered_x = x[Ax >= gamma]
    numerator = (filtered_Ax * filtered_x).sum()
    denominator = filtered_Ax.sum()
    print("numerator:", numerator)
    print("denominator:", denominator)
    return numerator / denominator


def indexOfRelationality(x, y):
    x, y = _transXAndY(x, y)
    rel = 0
    for l in range(len(x)):
        for k in range(l + 1, len(x)):
            delta_x = abs(x[k] - x[l])
            delta_y = abs(y[k] - y[l])
            if delta_y > delta_x:
                rel += 1 - delta_x / delta_y
    return rel


def mapTo(x, Ax, func):
    x, Ax = _transXAndAx(x, Ax)
    res = {}
    for a, b in zip(x, Ax):
        key = func(a)
        res.setdefault(key, [])
        res[key].append(b)
    for k, v in res.items():
        res[k] = max(v)
    return res


def orderedWeightedAverage(values, weights):
    assert len(values) == len(weights)
    sorted_values = sorted(values)
    return sum([i * j for i, j in zip(sorted_values, weights)])
