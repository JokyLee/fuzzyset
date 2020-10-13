#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.09.21'


import numpy as np
import numpy.testing as npTest

import fzset

from unittest import TestCase


class TestRelation(TestCase):
    def testMaxMinComposition(self):
        T_GT = np.array([
            [1, 1, 1, 1],
            [1, 1, 0, 1],
            [0, 0, 0, 0],
        ])
        R = np.array([
            [1, 1],
            [0, 1],
            [0, 0],
        ])
        G = np.array([
            [1, 0, 1, 1],
            [1, 1, 0, 1],
        ])
        T = fzset.supMinComposition(R, G)
        npTest.assert_allclose(T, T_GT)

    def testMinMaxComposition(self):
        T_GT = np.array([
            [1, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 0, 0, 1],
        ])
        R = np.array([
            [1, 1],
            [0, 1],
            [0, 0],
        ])
        G = np.array([
            [1, 0, 1, 1],
            [1, 1, 0, 1],
        ])
        T = fzset.infMaxComposition(R, G)
        npTest.assert_allclose(T, T_GT)
