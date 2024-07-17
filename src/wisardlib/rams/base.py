from numba import jit

from abc import abstractmethod

from wisardlib.config.type_definitions import ByteArray


@jit(nopython=True, inline="always")
def _encode_key(key: ByteArray) -> str:
    res = 0
    for i in range(len(key)):
        res += key[i] << i
    return str(res)
    # return str(sum(a<<i for i,a in enumerate(key)))


class RAM:
    """Random access memory implementation."""

    def encode_key(self, key: ByteArray):
        return str().join(str(k * 1) for k in key) # Used for large keys, when overflow
        # return _encode_key(key)
        # res = sum(a<<i for i,a in enumerate(key))
        # return str(res)

    @abstractmethod
    def add_member(self, key: ByteArray, inc_val: int = 1):
        raise NotImplementedError

    @abstractmethod
    def __contains__(self, key: ByteArray) -> bool:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, key: ByteArray) -> int:
        raise NotImplementedError

    def size(self):
        raise NotImplementedError
    
    def false_positive_rate(self) -> float:
        return 0


class JoinableRAM(RAM):
    def join(self, other: "JoinableRAM"):
        raise NotImplementedError
