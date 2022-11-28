import numpy as np
import numba.core.types
from numba.typed import Dict
from .base import RAM
from typing import Hashable


class DictRAM(RAM):
    def __init__(self):
        self._addresses = Dict.empty(
            key_type=numba.core.types.string, value_type=numba.core.types.int64
        )

    def encode_key(self, key: np.ndarray):
        return str().join(str(k * 1) for k in key)

    def add_member(self, key, inc_val: int = 1):
        key = self.encode_key(key)
        if key not in self._addresses:
            self._addresses[key] = inc_val
        else:
            self._addresses[key] += inc_val

    def __contains__(self, key):
        key = self.encode_key(key)
        return key in self._addresses

    def __getitem__(self, key):
        key = self.encode_key(key)
        return self._addresses.get(key, 0)

    def __str__(self) -> str:
        return f"DictRAM with {len(self._addresses)} addresses."

    def __repr__(self) -> str:
        return str(self)
