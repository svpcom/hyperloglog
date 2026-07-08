"""
Sliding HyperLogLog
"""

import math
import heapq
import numpy as np

from hashlib import sha1
from msgpack import packb
from .hll import get_alpha, get_rho, get_estimate


class SlidingHyperLogLog(object):
    """
    Sliding HyperLogLog: Estimating cardinality in a data stream (Telecom ParisTech)
    """

    __slots__ = ('window', 'alpha', 'p', 'm', 'LPFM')

    def __init__(self, error_rate, window, lpfm=None):
        """
        Implementes a Sliding HyperLogLog

        error_rate = abs_err / cardinality
        """

        self.window = window

        if not (window > 0):
            raise ValueError('window must be > 0')

        if lpfm is not None:
            m = len(lpfm)

            if m == 0 or (m & (m - 1)) != 0:
                raise ValueError('List length is not power of 2')

            p = m.bit_length() - 1
            self.LPFM = list(lpfm)

        else:
            if not (0 < error_rate < 1):
                raise ValueError("Error_Rate must be between 0 and 1.")

            # error_rate = 1.04 / sqrt(m)
            # m = 2 ** p

            p = math.ceil(math.log((1.04 / error_rate) ** 2, 2))
            m = 1 << p
            self.LPFM = [tuple() for i in range(m)]

        self.alpha = get_alpha(p)
        self.p = p
        self.m = m

    def __getstate__(self):
        return dict([x, getattr(self, x)] for x in self.__slots__)

    def __setstate__(self, d):
        for key in d:
            setattr(self, key, d[key])

        # normalize LPFM from pickles created by older versions, which used
        # None for empty registers and ascending timestamp order
        self.LPFM = [tuple(sorted(reg, reverse=True)) if reg else tuple() for reg in self.LPFM]

    @classmethod
    def from_list(cls, lpfm, window):
        return cls(None, window, lpfm)

    def add(self, timestamp, value):
        """
        Adds the item to the HyperLogLog
        """
        # h: D -> {0,1} ** 64
        # x = h(v)
        # j = <x_0x_1..x_{p-1})>
        # w = <x_{p}x_{p+1}..>
        # <t_i, rho(w)>

        x = int.from_bytes(sha1(packb(value)).digest()[:8], byteorder='big')
        j = x & (self.m - 1)
        w = x >> self.p
        R = get_rho(w, 64 - self.p)

        Rmax = None
        tmp = []
        tmax = None

        for t, R in heapq.merge(self.LPFM[j], ((timestamp, R),), reverse=True):
            if tmax is None:
                tmax = t - self.window

            if t < tmax:
                break

            if Rmax is None or R > Rmax:
                tmp.append((t, R))
                Rmax = R

        self.LPFM[j] = tuple(tmp)

    def update(self, *others):
        """
        Merge other counters
        """

        for item in others:
            if self.m != item.m:
                raise ValueError('Counters precisions should be equal')

            if self.window != item.window:
                raise ValueError('Counters windows should be equal')

        for j, lpfms_j in enumerate(zip(self.LPFM, *list(item.LPFM for item in others))):
            Rmax = None
            tmp = []
            tmax = None

            for t, R in heapq.merge(*lpfms_j, reverse=True):
                if tmax is None:
                    tmax = t

                if t < (tmax - self.window):
                    break

                if Rmax is None or R > Rmax:
                    tmp.append((t, R))
                    Rmax = R

            self.LPFM[j] = tuple(tmp)

    def __eq__(self, other):
        if not isinstance(other, SlidingHyperLogLog):
            return NotImplemented

        return self.m == other.m and self.window == other.window and self.LPFM == other.LPFM

    def __len__(self):
        raise NotImplementedError('use card(timestamp) to estimate cardinality')

    def card(self, timestamp, window=None):
        """
        Returns the estimate of the cardinality at 'timestamp' using 'window'
        """
        if window is None:
            window = self.window

        if not 0 < window <= self.window:
            raise ValueError('0 < window <= W')

        _t = timestamp - window
        M = np.fromiter((np.max(np.fromiter((R for ts, R in lpfm if ts >= _t), int), initial=0) if lpfm else 0 for lpfm in self.LPFM), int)

        return get_estimate(M, self.m, self.p, self.alpha)

    def card_wlist(self, timestamp, window_list):
        """
        Returns the estimate of the cardinality at 'timestamp' using list of windows
        """
        for window in window_list:
            if not 0 < window <= self.window:
                raise ValueError('0 < window <= W')

        tsl = sorted((timestamp - window, idx) for idx, window in enumerate(window_list))
        M_list = [[] for _ in window_list]

        for lpfm in self.LPFM:
            R_max = 0
            _p = len(tsl) - 1

            for ts, R in lpfm:
                while _p >= 0:
                    _ts, _idx = tsl[_p]
                    if ts >= _ts: break
                    M_list[_idx].append(R_max)
                    _p -= 1
                if _p < 0: break
                R_max = R

            for i in range(0, _p + 1):
                M_list[tsl[i][1]].append(R_max)

        res = []
        for M in M_list:
            res.append(get_estimate(np.array(M, int), self.m, self.p, self.alpha))
        return res
