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


class TestAHP(TestCase):
    def testR(self):
        R = np.array([
            [1, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 7, 1 / 9],
            [2, 1, 1 / 2, 1 / 3, 1 / 4, 1 / 6, 1 / 8],
            [3, 2, 1, 1 / 2, 1 / 3, 1 / 4, 1 / 5],
            [4, 3, 2, 1, 1 / 2, 1 / 2, 1 / 3],
            [5, 4, 3, 2, 1, 1 / 2, 1 / 2],
            [7, 6, 4, 2, 2, 1, 1 / 2],
            [9, 8, 5, 3, 2, 2, 1],
        ])
        l_max, membership = fzset.analyticHierarchyProcess(R)
        npTest.assert_allclose(l_max, 7.1, atol=0.01)
        npTest.assert_allclose(
            membership, [0.08, 0.12, 0.19, 0.33, 0.49, 0.7, 1.0], atol=0.01
        )
