#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.09.18'
__all__ = [
    "Interval",
]


import itertools


class Interval:
    def __init__(self, interval):
        assert isinstance(interval, (list, tuple))
        assert len(interval) == 2
        self._interval = tuple(interval)

    @property
    def min(self):
        return self._interval[0]

    @property
    def max(self):
        return self._interval[1]

    def __str__(self):
        return self._interval.__str__()

    def __add__(self, other):
        assert isinstance(other, Interval)
        lower = self.min + other.min
        upper = self.max + other.max
        return Interval((lower, upper))

    def __mul__(self, other):
        if isinstance(other, Interval):
            candidates = [a * b for (a, b) in itertools.product(self._interval, other._interval)]
            return Interval((min(candidates), max(candidates)))
        elif isinstance(other, (int, float)):
            lower = self.min * other
            upper = self.max * other
            return Interval((lower, upper))
        else:
            raise ValueError("must be (Interval, int, float)")

    def __sub__(self, other):
        assert isinstance(other, Interval)
        return self + Interval((-1, -1)) * other

    def __truediv__(self, other):
        assert isinstance(other, Interval)
        return self * Interval((1 / other.max, 1 / other.min))
