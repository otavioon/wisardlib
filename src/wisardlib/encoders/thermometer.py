import numpy as np

# @jit(nopython=True, inline='always')
def int_to_binary_list(value: int, size: int, reverse: bool = False):
    if not reverse:
        return [(value >> i) % 2 for i in range(size)]
    else:
        return [(value >> i) % 2 for i in range(size, 0, -1)]


class ThermometerEncoder:
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

    def transform(self, X: np.ndarray):
        new_shape = X.shape + (self.resolution,)
        new_X = [
            int_to_binary_list((1 << x + 1) - 1, self.resolution, reverse=True)
            for x in np.digitize(X - self.min_val, self.buckets).ravel()
        ]
        return np.array(new_X, dtype=bool).reshape(new_shape)

    def fit_transform(self, X, y=None, **fit_args):
        return self.fit(X, y, **fit_args).transform(X)


class NestedThermometerEncoder:
    def __init__(self, resolution: int, resolution_2: int):
        self.resolution = resolution
        self.resolution_2 = resolution_2
        self.thermometer = ThermometerEncoder(resolution_2)
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

    def transform(self, X: np.ndarray):
        dig_X = np.digitize(X - self.min_val, self.buckets)
        return self.thermometer.fit_transform(X)

    def fit_transform(self, X, y=None, **fit_args):
        return self.fit(X, y, **fit_args).transform(X)
