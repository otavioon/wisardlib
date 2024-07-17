from numba import jit

from abc import abstractmethod

from wisardlib.config.type_definitions import BooleanArray
from numba import jit
from numba import jit
import numpy as np

@jit(nopython=True)
def _encode_key(key: np.ndarray) -> str:
    return ''.join(map(str, key))


class RAM:
    """Random access memory implementation."""

    def encode_key(self, key: BooleanArray):
        return _encode_key(key)
    
        # res = sum(a<<i for i,a in enumerate(key))
        # return str(res)

        # return str().join(str(k * 1) for k in key) # Used for large keys, when overflow

    @abstractmethod
    def add_member(self, key: BooleanArray, inc_val: int = 1):
        raise NotImplementedError

    @abstractmethod
    def __contains__(self, key: BooleanArray) -> bool:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, key: BooleanArray) -> int:
        raise NotImplementedError

    def size(self):
        raise NotImplementedError
    
    def false_positive_rate(self) -> float:
        return 0


class JoinableRAM(RAM):
    def join(self, other: "JoinableRAM"):
        raise NotImplementedError
