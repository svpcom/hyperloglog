"""
Sliding HyperLogLog
"""

import math
import heapq
import struct
from hashlib import sha1
from .hll import get_treshold, estimate_bias, get_alpha, get_rho
from .compat import *


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
            p = int(round(math.log(m, 2)))

            if (1 << p) != m:
                raise ValueError('List length is not power of 2')
            self.LPFM = lpfm

        else:
            if not (0 < error_rate < 1):
                raise ValueError("Error_Rate must be between 0 and 1.")

            # error_rate = 1.04 / sqrt(m)
            # m = 2 ** p

            p = int(math.ceil(math.log((1.04 / error_rate) ** 2, 2)))
            m = 1 << p
            self.LPFM = [None for i in range(m)]

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

        if isinstance(value, unicode):
            value = value.encode('utf-8')
        elif not isinstance(value, bytes):
            value = bytes(value)

        x = struct.unpack('!Q', sha1(value).digest()[:8])[0]
        j = x & (self.m - 1)
        w = x >> self.p
        R = get_rho(w, 64 - self.p)

        Rmax = None
        tmp = []
        tmax = None
        tmp2 = list(heapq.merge(self.LPFM[j] if self.LPFM[j] is not None else [], [(timestamp, R)]))

        for t, R in reversed(tmp2):
            if tmax is None:
                tmax = t

            if t < (tmax - self.window):
                break

            if Rmax is None or R > Rmax:
                tmp.append((t, R))
                Rmax = R

        tmp.reverse()
        self.LPFM[j] = tuple(tmp) if tmp else None

    def update(self, *others):
        """
        Merge other counters
        """

        for item in others:
            if self.m != item.m:
                raise ValueError('Counters precisions should be equal')

        for j in xrange(len(self.LPFM)):
            Rmax = None
            tmp = []
            tmax = None
            tmp2 = list(heapq.merge(*([item.LPFM[j] if item.LPFM[j] is not None else [] for item in others] + [self.LPFM[j] if self.LPFM[j] is not None else []])))

            for t, R in reversed(tmp2):
                if tmax is None:
                    tmax = t

                if t < (tmax - self.window):
                    break

                if Rmax is None or R > Rmax:
                    tmp.append((t, R))
                    Rmax = R

            tmp.reverse()
            self.LPFM[j] = tuple(tmp) if tmp else None

    def __eq__(self, other):
        if self.m != other.m:
            raise ValueError('Counters precisions should be equal')

        return self.LPFM == other.LPFM

    def __ne__(self, other):
        return not self.__eq__(other)

    def __len__(self):
        raise NotImplemented

    def _Ep(self, M):
        E = self.alpha * float(self.m ** 2) / sum(math.pow(2.0, -x) for x in M)
        return (E - estimate_bias(E, self.p)) if E <= 5 * self.m else E

    def card(self, timestamp, window=None):
        """
        Returns the estimate of the cardinality at 'timestamp' using 'window'
        """
        if window is None:
            window = self.window

        if not 0 < window <= self.window:
            raise ValueError('0 < window <= W')

        def max_r(l):
            return max(l) if l else 0

        M = [max_r([R for ts, R in lpfm if ts >= (timestamp - window)]) if lpfm else 0 for lpfm in self.LPFM]

        #count number or registers equal to 0
        V = M.count(0)
        if V > 0:
            H = self.m * math.log(self.m / float(V))
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

        tsl = [(timestamp - window, idx) for idx, window in enumerate(window_list)]
        tsl.sort()

        M_list = [[] for _ in window_list]

        # Highly optimized code (PyPy), but may be slow in CPython
        for lpfm in self.LPFM:
            R_max = 0
            _p = len(tsl) - 1
            if lpfm:
                i = len(lpfm) - 1
                while i >= 0:
                    ts, R = lpfm[i]
                    while _p >= 0:
                        _ts, _idx = tsl[_p]
                        if ts >= _ts: break
                        M_list[_idx].append(R_max)
                        _p -= 1
                    if _p < 0: break
                    R_max = R
                    i -= 1

            for i in xrange(0, _p + 1):
                M_list[tsl[i][1]].append(R_max)

        res = []
        for M in M_list:
            #count number or registers equal to 0
            V = M.count(0)
            if V > 0:
                H = self.m * math.log(self.m / float(V))
                res.append(H if H <= get_treshold(self.p) else self._Ep(M))
            else:
                res.append(self._Ep(M))
        return res
