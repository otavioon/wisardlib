from numba import jit

from abc import abstractmethod

from wisardlib.config.type_definitions import BooleanArray


@jit(nopython=True, inline="always")
def _encode_key(key: BooleanArray) -> str:
    res = 0
    for i in range(len(key)):
        res += key[i] << i
    return str(res)
    # return str(sum(a<<i for i,a in enumerate(key)))


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
