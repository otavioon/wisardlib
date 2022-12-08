import numpy as np
from wisardlib.config.type_definitions import BooleanArray


class Encoder:
    """Encode a numpy array into a array of booleans."""

    def fit(self, X: np.ndarray, y=None, **fit_args):
        return self

    def transform(self, X: np.ndarray) -> BooleanArray:
        raise NotImplementedError

    def fit_transform(self, X: np.ndarray, y=None, **fit_args) -> BooleanArray:
        return fit(X, y, **fit_args)


class EncoderDecoder(Encoder):
    def inverse_transform(self, X: BooleanArray) -> np.ndarray:
        raise NotImplementedError
