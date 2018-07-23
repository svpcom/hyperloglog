#!/usr/bin/env python

import os
import math
import time
from unittest import TestCase
from hyperloglog.shll import SlidingHyperLogLog
from hyperloglog.compat import *
import pickle


class SlidingHyperLogLogTestCase(TestCase):
    def test_init(self):
        s = SlidingHyperLogLog(0.05, 100)
        self.assertEqual(s.window, 100)
        self.assertEqual(s.p, 9)
        self.assertEqual(s.alpha, 0.7197831133217303)
        self.assertEqual(s.m, 512)
        self.assertEqual(len(s.LPFM), 512)

    def test_add(self):
        s = SlidingHyperLogLog(0.05, 100)

        for i in range(10):
            s.add(i, str(i))

        M = [(i, max(R for ts, R in lpfm)) for i, lpfm in enumerate(s.LPFM) if lpfm]
        self.assertEqual(M, [(1, 1), (41, 1), (44, 1), (76, 3), (103, 4), (182, 1), (442, 2), (464, 5), (497, 1), (506, 1)])

    def test_from_list(self):
        s1 = SlidingHyperLogLog(0.05, 100)

        for i in range(10):
            s1.add(i, str(i))

        s2 = SlidingHyperLogLog.from_list(s1.LPFM, 100)
        self.assertEqual(s1, s2)
        self.assertEqual(s1.card(9), s2.card(9))
        self.assertEqual(s1.card_wlist(9, [100, 3, 5]), [s2.card(9, 100), s2.card(9, 3), s2.card(9, 5)])

    def test_calc_cardinality(self):
        clist = [1, 5, 10, 30, 60, 200, 1000, 10000, 60000]
        n = 30
        rel_err = 0.05

        for card in clist:
            s = 0.0
            for c in xrange(n):
                a = SlidingHyperLogLog(rel_err, 100)

                for i in xrange(card):
                    a.add(int(time.time()), os.urandom(20))

                s += a.card(int(time.time()))

            z = (float(s) / n - card) / (rel_err * card / math.sqrt(n))
            self.assertLess(-3, z)
            self.assertGreater(3, z)

    def test_calc_cardinality_sliding1(self):
        a = SlidingHyperLogLog(0.05, 100)
        a.add(1, 'k1')
        self.assertEqual(int(a.card(1)), 1)
        self.assertEqual(int(a.card(101)), 1)
        self.assertEqual(int(a.card(102)), 0)
        a.add(2, 'k2')
        a.add(3, 'k3')
        self.assertEqual(int(a.card(3)), 3)
        self.assertEqual(int(a.card(101)), 3)
        self.assertEqual(int(a.card(102)), 2)
        self.assertEqual(int(a.card(103)), 1)
        self.assertEqual(int(a.card(104)), 0)

    def test_calc_cardinality_sliding2(self):
        clist = [1, 5, 10, 30, 60, 200, 1000, 10000, 60000]
        n = 30
        rel_err = 0.05

        for card in clist:
            s = 0.0
            for c in xrange(n):
                a = SlidingHyperLogLog(rel_err, 100)

                for i in xrange(card):
                    a.add(i / 2000.0, os.urandom(20))

                s += a.card(card / 2000.0)

            card_stored = min(card, 200000)
            z = (float(s) / n - card_stored) / (rel_err * card_stored / math.sqrt(n))
            self.assertLess(-3, z)
            self.assertGreater(3, z)

    def test_calc_cardinality_sliding3(self):
        clist = [30, 60, 200, 1000, 10000, 60000]
        rel_err = 0.05
        t1 = 0
        t2 = 0
        for card in clist:
            a = SlidingHyperLogLog(rel_err, card)

            for i in xrange(card):
                a.add(i, os.urandom(20))

            ts = time.time()
            l1 = [a.card(1.5 * card, w / 10.0) for w in range(1, card + 1, card // 10)]
            t1 = (time.time() - ts)
            ts = time.time()
            l2 = a.card_wlist(1.5 * card, [ w / 10.0 for w in range(1, card + 1, card // 10)])
            t2 = (time.time() - ts)
            # print card, t1, t2
            self.assertEqual(l1, l2)

    def test_update(self):
        a = SlidingHyperLogLog(0.05, 100)
        b = SlidingHyperLogLog(0.05, 100)
        c = SlidingHyperLogLog(0.05, 100)

        for i in xrange(10000):
            a.add(i, str('k1-%d' % i))
            c.add(i, str('k1-%d' % i))

        for i in xrange(10000):
            b.add(i, str('k2-%d' % i))
            c.add(i, str('k2-%d' % i))

        a.update(b)

        self.assertNotEqual(a, b)
        self.assertNotEqual(b, c)
        self.assertEqual(a, c)

    def test_update_err(self):
        a = SlidingHyperLogLog(0.05, 100)
        b = SlidingHyperLogLog(0.01, 100)

        self.assertRaises(ValueError, a.update, b)

    def test_pickle(self):
        a = SlidingHyperLogLog(0.05, 100)
        for i in xrange(10000):
            a.add(i, str('k1-%d' % i))
        
        b = pickle.loads(pickle.dumps(a))
        self.assertEqual(a.window, b.window)
        self.assertEqual(a.alpha, b.alpha)
        self.assertEqual(a.p, b.p)
        self.assertEqual(a.m, b.m)
        self.assertEqual(a.LPFM, b.LPFM)
