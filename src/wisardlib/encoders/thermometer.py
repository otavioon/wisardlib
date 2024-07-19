import logging
import time

import numpy as np
import pandas as pd

from wisardlib.config.type_definitions import ByteArray
from wisardlib.encoders.base import Encoder

class ThermometerEncoder(Encoder):
    def __init__(self, resolution: int):
        self.resolution = resolution
        self.min_val = None
        self.buckets = None

    def fit(self, X, y=None, **fit_args):
        self.min_val = np.min(X)
        max_val = np.max(X)
        self.buckets = ((np.arange(self.resolution) + 1) * (max_val - self.min_val) / (self.resolution + 1))
        return self

    def transform(self, X: np.ndarray) -> ByteArray:
        new_shape = X.shape + (self.resolution,)
        digitized = np.digitize(X - self.min_val, self.buckets).ravel()
        
        # Calculate the binary representation in one step
        new_X = (digitized[:, None] > np.arange(self.resolution)).astype(bool)
        
        # Reverse the values
        new_X = new_X[:, ::-1]

        return new_X.reshape(new_shape)


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
