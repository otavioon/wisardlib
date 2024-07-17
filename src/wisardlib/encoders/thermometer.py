import logging

import numpy as np
import pandas as pd

from wisardlib.config.type_definitions import ByteArray
from wisardlib.encoders.base import Encoder

# @jit(nopython=True, inline='always')
def int_to_binary_list(value: int, size: int, reverse: bool = False):
    if not reverse:
        return [(value >> i) % 2 for i in range(size)]
    else:
        return [(value >> i) % 2 for i in range(size, 0, -1)]


class ThermometerEncoder(Encoder):
    def __init__(self, resolution: int):
        self.resolution = resolution
        self.min_val = None
        self.buckets = []

    def fit(self, X, y=None, **fit_args):
        self.min_val = np.min(X)
        self.buckets = (
            (np.arange(self.resolution) + 1)
            * (np.max(X) - np.min(X))
            / (self.resolution + 1)
        )
        return self

    def transform(self, X: np.ndarray) -> ByteArray:
        new_shape = X.shape + (self.resolution,)
        new_X = [
            int_to_binary_list((1 << x + 1) - 1, self.resolution, reverse=True)
            for x in np.digitize(X - self.min_val, self.buckets).ravel()
        ]
        return np.array(new_X, dtype=bool).reshape(new_shape)


class DistributiveThermometerEncoder(ThermometerEncoder):
    def fit(self, X, y=None, **fit_args):
        self.min_val = np.min(X)
        res_min = self.resolution
        res_max = self.resolution * 10
        last_q = 0

        while res_min < res_max:
            q = (res_max + res_min) // 2
            result = pd.qcut(X.ravel(), q=q, duplicates="drop")
            size = len(result.categories)
            # print(f"q: {q}, size: {size}, min: {res_min}, max: {res_max}")

            if size == self.resolution:
                self.buckets = [x.right for x in result.categories]
                return self

            if size > self.resolution:
                res_max = q
            else:
                res_min = q

            # Not needed
            if last_q == q:
                break
            else:
                last_q = q

        raise ValueError("Could not find a valid split")
