#!/usr/bin/env python

from unittest import TestCase
from hyperloglog.hll import HyperLogLog, get_alpha, get_rho
from hyperloglog.const import biasData, tresholdData, rawEstimateData
from hyperloglog.compat import *
import math
import os
import pickle


class HyperLogLogTestCase(TestCase):
    def test_blobs(self):
        self.assertEqual(len(tresholdData), 18 - 3)

    def test_alpha(self):
        alpha = [get_alpha(b) for b in range(4, 10)]
        self.assertEqual(alpha, [0.673, 0.697, 0.709, 0.7152704932638152, 0.7182725932495458, 0.7197831133217303])

    def test_alpha_bad(self):
        self.assertRaises(ValueError, get_alpha, 1)
        self.assertRaises(ValueError, get_alpha, 17)

    def test_rho(self):
        self.assertEqual(get_rho(0, 32), 33)
        self.assertEqual(get_rho(1, 32), 32)
        self.assertEqual(get_rho(2, 32), 31)
        self.assertEqual(get_rho(3, 32), 31)
        self.assertEqual(get_rho(4, 32), 30)
        self.assertEqual(get_rho(5, 32), 30)
        self.assertEqual(get_rho(6, 32), 30)
        self.assertEqual(get_rho(7, 32), 30)
        self.assertEqual(get_rho(1 << 31, 32), 1)
        self.assertRaises(ValueError, get_rho, 1 << 32, 32)

    def test_rho_emu(self):
        from hyperloglog import hll
        old = hll.bit_length
        hll.bit_length = hll.bit_length_emu
        try:
            self.assertEqual(get_rho(0, 32), 33)
            self.assertEqual(get_rho(1, 32), 32)
            self.assertEqual(get_rho(2, 32), 31)
            self.assertEqual(get_rho(3, 32), 31)
            self.assertEqual(get_rho(4, 32), 30)
            self.assertEqual(get_rho(5, 32), 30)
            self.assertEqual(get_rho(6, 32), 30)
            self.assertEqual(get_rho(7, 32), 30)
            self.assertEqual(get_rho(1 << 31, 32), 1)
            self.assertRaises(ValueError, get_rho, 1 << 32, 32)
        finally:
            hll.bit_length = old

    def test_init(self):
        s = HyperLogLog(0.05)
        self.assertEqual(s.p, 9)
        self.assertEqual(s.alpha, 0.7197831133217303)
        self.assertEqual(s.m, 512)
        self.assertEqual(len(s.M), 512)

    def test_add(self):
        s = HyperLogLog(0.05)

        for i in range(10):
            s.add(str(i))

        M = [(i, v) for i, v in enumerate(s.M) if v > 0]

        self.assertEqual(M, [(1, 1), (41, 1), (44, 1), (76, 3), (103, 4), (182, 1), (442, 2), (464, 5), (497, 1), (506, 1)])

    def test_calc_cardinality(self):
        clist = [1, 5, 10, 30, 60, 200, 1000, 10000, 60000]
        n = 30
        rel_err = 0.05

        for card in clist:
            s = 0.0
            for c in xrange(n):
                a = HyperLogLog(rel_err)

                for i in xrange(card):
                    a.add(os.urandom(20))

                s += a.card()

            z = (float(s) / n - card) / (rel_err * card / math.sqrt(n))
            self.assertLess(-3, z)
            self.assertGreater(3, z)


    def test_update(self):
        a = HyperLogLog(0.05)
        b = HyperLogLog(0.05)
        c = HyperLogLog(0.05)

        for i in xrange(2):
            a.add(str(i))
            c.add(str(i))

        for i in xrange(2, 4):
            b.add(str(i))
            c.add(str(i))

        a.update(b)

        self.assertNotEqual(a, b)
        self.assertNotEqual(b, c)
        self.assertEqual(a, c)


    def test_update_err(self):
        a = HyperLogLog(0.05)
        b = HyperLogLog(0.01)

        self.assertRaises(ValueError, a.update, b)

    def test_pickle(self):
        a = HyperLogLog(0.05)
        for x in range(100):
            a.add(str(x))
        b = pickle.loads(pickle.dumps(a))
        self.assertEqual(a.M, b.M)
        self.assertEqual(a.alpha, b.alpha)
        self.assertEqual(a.p, b.p)
        self.assertEqual(a.m, b.m)
