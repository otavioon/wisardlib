import numpy as np
from wisardlib.config.type_definitions import ByteArray


class Encoder:
    """Encode a numpy array into a array of booleans."""

    def fit(self, X: np.ndarray, y=None, **fit_args):
        return self

    def transform(self, X: np.ndarray) -> ByteArray:
        raise NotImplementedError

    def fit_transform(self, X: np.ndarray, y=None, **fit_args) -> ByteArray:
        return self.fit(X, y, **fit_args).transform(X)


class EncoderDecoder(Encoder):
    def inverse_transform(self, X: ByteArray) -> np.ndarray:
        raise NotImplementedError
