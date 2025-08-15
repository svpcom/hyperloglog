"""
This module implements probabilistic data structure which is able to calculate the cardinality of large multisets in a single pass using little auxiliary memory
"""

import math
import numpy as np
from hashlib import sha1
from msgpack import packb

from .const import rawEstimateData as _rawEstimateData, \
                   biasData as _biasData, \
                   tresholdData as _tresholdData

rawEstimateData = tuple(np.array(row) for row in _rawEstimateData)
biasData = tuple(np.array(row) for row in _biasData)
tresholdData = tuple(np.array(row) for row in _tresholdData)


def get_treshold(p):
    return tresholdData[p - 4]


def estimate_bias(E, p):
    bias_vector = biasData[p - 4]
    nearest_neighbors = get_nearest_neighbors(E, rawEstimateData[p - 4])
    return np.sum(bias_vector[nearest_neighbors]) / len(nearest_neighbors)


def get_nearest_neighbors(E, estimate_vector):
    return np.argsort((E - estimate_vector) ** 2)[:6]


def get_alpha(p):
    if not (4 <= p <= 16):
        raise ValueError("p=%d should be in range [4 : 16]" % p)

    if p == 4:
        return 0.673

    if p == 5:
        return 0.697

    if p == 6:
        return 0.709

    return 0.7213 / (1.0 + 1.079 / (1 << p))


def get_rho(w, max_width):
    rho = max_width - w.bit_length() + 1

    if rho <= 0:
        raise ValueError('w overflow')

    return rho


# Check for NumPy 2.x
if hasattr(np,'bitwise_count'):
    def bit_length_vec(arr):
        bits = arr >> 1
        bits |= arr
        bits |= bits >> 2
        bits |= bits >> 4
        bits |= bits >> 8
        bits |= bits >> 16
        bits |= bits >> 32
        return np.bitwise_count(bits)
else:
    def bit_length_vec(arr):
        _, high_exp = np.frexp(arr >> 32)
        _, low_exp = np.frexp(arr & 0xFFFFFFFF)
        return np.where(high_exp, high_exp + 32, low_exp)


def get_rho_vec(w, max_width):
    rho = max_width - bit_length_vec(w) + 1

    if np.count_nonzero(rho <= 0):
        raise ValueError('w overflow')

    return rho


class HyperLogLog(object):
    """
    HyperLogLog cardinality counter
    """

    __slots__ = ('alpha', 'p', 'm', 'M')

    def __init__(self, error_rate):
        """
        Implementes a HyperLogLog

        error_rate = abs_err / cardinality
        """

        if not (0 < error_rate < 1):
            raise ValueError("Error_Rate must be between 0 and 1.")

        # error_rate = 1.04 / sqrt(m)
        # m = 2 ** p
        # M(1)... M(m) = 0

        p = math.ceil(math.log((1.04 / error_rate) ** 2, 2))

        self.alpha = get_alpha(p)
        self.p = p
        self.m = 1 << p
        self.M = np.zeros(self.m, np.int8)

    def __getstate__(self):
        return dict([x, getattr(self, x)] for x in self.__slots__)

    def __setstate__(self, d):
        for key in d:
            setattr(self, key, d[key])

    def add(self, value):
        """
        Adds the item to the HyperLogLog
        """
        # h: D -> {0,1} ** 64
        # x = h(v)
        # j = <x_0x_1..x_{p-1}>
        # w = <x_{p}x_{p+1}..>
        # M[j] = max(M[j], rho(w))

        x = int.from_bytes(sha1(packb(value)).digest()[:8], byteorder='big')
        j = x & (self.m - 1)
        w = x >> self.p
        rho = get_rho(w, 64 - self.p)

        self.M[j] = max(self.M[j], rho)

    def add_bulk(self, values):
        """
        Adds the item to the HyperLogLog
        """
        # h: D -> {0,1} ** 64
        # x = h(v)
        # j = <x_0x_1..x_{p-1}>
        # w = <x_{p}x_{p+1}..>
        # M[j] = max(M[j], rho(w))

        assert not isinstance(values, (bytes, str)) and hasattr(values, '__iter__')

        x = np.fromiter((int.from_bytes(sha1(packb(value)).digest()[:8], byteorder='big') for value in values), np.uint64)
        j = x & (self.m - 1)
        w = x >> self.p
        rho = get_rho_vec(w, 64 - self.p)

        unique_j, inverse_indices = np.unique(j, return_inverse=True)
        sort_inv = np.argsort(inverse_indices)
        group_start_indices = np.searchsorted(inverse_indices[sort_inv], np.arange(len(unique_j)))
        grouped_rho = np.maximum.reduceat(rho[sort_inv], group_start_indices)

        self.M[unique_j] = np.maximum(self.M[unique_j], grouped_rho)

    def update(self, *others):
        """
        Merge other counters
        """
        ml = [self.M]

        for item in others:
            if self.m != item.m:
                raise ValueError('Counters precisions should be equal')
            ml.append(item.M)

        self.M = np.maximum.reduce(ml)

    def __eq__(self, other):
        if self.m != other.m:
            raise ValueError('Counters precisions should be equal')

        return np.array_equal(self.M, other.M)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __len__(self):
        return round(self.card())

    def _Ep(self):
        E = self.alpha * (self.m ** 2) / np.power(2.0, -self.M, dtype=float).sum()
        return (E - estimate_bias(E, self.p)) if E <= 5 * self.m else E

    def card(self):
        """
        Returns the estimate of the cardinality
        """

        #count number of registers equal to 0
        V = np.count_nonzero(self.M == 0)

        if V > 0:
            H = self.m * math.log(self.m / V)
            return H if H <= get_treshold(self.p) else self._Ep()
        else:
            return self._Ep()

