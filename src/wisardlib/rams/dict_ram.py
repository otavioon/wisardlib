import pickle
import tempfile
from pathlib import Path

import numpy as np
import numba.core.types
from numba.typed import Dict
from .base import JoinableRAM
from typing import Hashable


from wisardlib.config.type_definitions import BooleanArray


def size_of_dict(numba_dict) -> int:
    with tempfile.NamedTemporaryFile() as tmp:
        pickle.dump(dict(numba_dict), tmp, pickle.HIGHEST_PROTOCOL)
        tmp.flush()
        return Path(tmp.name).stat().st_size


class DictRAM(JoinableRAM):
    def __init__(self):
        self._addresses = Dict.empty(
            key_type=numba.core.types.string, value_type=numba.core.types.int64
        )
        self._key_len = 1

    # def encode_key(self, key: BooleanArray):
    #     k = str().join(str(k * 1) for k in key)
    #     self._key_len = len(k)
    #     return k

    def add_member(self, key: BooleanArray, inc_val: int = 1):
        key = self.encode_key(key)
        if key not in self._addresses:
            self._addresses[key] = inc_val
        else:
            self._addresses[key] += inc_val

    def join(self, other: "DictRAM"):
        for key, value in other._addresses.items():
            self._addresses[key] = self._addresses.get(key, 0) + value

    def __contains__(self, key: BooleanArray):
        key = self.encode_key(key)
        return key in self._addresses

    def __getitem__(self, key: BooleanArray):
        key = self.encode_key(key)
        return self._addresses.get(key, 0)

    def __str__(self) -> str:
        return f"DictRAM with {len(self._addresses)} addresses."

    def __repr__(self) -> str:
        return str(self)

    def size(self) -> int:
        return size_of_dict(self._addresses)
