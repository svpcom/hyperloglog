"""
Sliding HyperLogLog
"""

import math
import heapq
import numpy as np

from hashlib import sha1
from msgpack import packb
from .hll import get_treshold, estimate_bias, get_alpha, get_rho


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

        if lpfm is not None:
            m = len(lpfm)
            p = round(math.log(m, 2))

            if (1 << p) != m:
                raise ValueError('List length is not power of 2')

            self.LPFM = lpfm

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
        if self.m != other.m:
            raise ValueError('Counters precisions should be equal')

        return self.LPFM == other.LPFM

    def __ne__(self, other):
        return not self.__eq__(other)

    def __len__(self):
        raise NotImplemented

    def _Ep(self, M):
        E = self.alpha * (self.m ** 2) / np.power(2.0, -M, dtype=float).sum()
        return (E - estimate_bias(E, self.p)) if E <= 5 * self.m else E

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

        #count number or registers equal to 0
        V = np.count_nonzero(M == 0)

        if V > 0:
            H = self.m * math.log(self.m / V)
            return H if H <= get_treshold(self.p) else self._Ep(M)
        else:
            return self._Ep(M)

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
            #count number of registers equal to 0
            V = M.count(0)
            M = np.array(M, int)

            if V > 0:
                H = self.m * math.log(self.m / V)
                res.append(H if H <= get_treshold(self.p) else self._Ep(M))
            else:
                res.append(self._Ep(M))
        return res
