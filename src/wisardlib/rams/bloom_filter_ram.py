import numpy as np
import numba.core.types
from numba.typed import Dict
from .base import RAM
from typing import Hashable
from probables import CountMinSketch, CountingCuckooFilter, HeavyHitters, StreamThreshold


class CountMinSketchRAM(RAM):
    def __init__(
        self,
        width: int = 1000,
        depth: int = 5,
        confidence: float = None,
        soft_error_rate: float = None,
    ):
        self.bloom_filter = CountMinSketch(
            width=width, depth=depth, confidence=confidence, error_rate=soft_error_rate
        )

    def encode_key(self, key: np.ndarray):
        return str().join(str(k * 1) for k in key)

    def add_member(self, key, inc_val: int = 1):
        key = self.encode_key(key)
        self.bloom_filter.add(key)

    def __contains__(self, key):
        key = self.encode_key(key)
        return self.bloom_filter.check(key) > 0

    def __getitem__(self, key):
        key = self.encode_key(key)
        return self.bloom_filter.check(key)

    def __str__(self) -> str:
        return f"CountMinSketchRAM with {str(self.bloom_filter)}"

    def __repr__(self) -> str:
        return str(self)


class CountingCuckooRAM(RAM):
    def __init__(
        self,
        capacity=10000,
        bucket_size=4,
        max_swaps=500,
        expansion_rate=2,
        auto_expand=True,
        finger_size=4,
    ):
        self.bloom_filter = CountingCuckooFilter(
            capacity=capacity,
            bucket_size=bucket_size,
            expansion_rate=expansion_rate,
            auto_expand=auto_expand,
            finger_size=finger_size,
        )

    def encode_key(self, key: np.ndarray):
        return str().join(str(k * 1) for k in key)

    def add_member(self, key, inc_val: int = 1):
        key = self.encode_key(key)
        self.bloom_filter.add(key)

    def __contains__(self, key):
        key = self.encode_key(key)
        return self.bloom_filter.check(key) > 0

    def __getitem__(self, key):
        key = self.encode_key(key)
        return self.bloom_filter.check(key)

    def __str__(self) -> str:
        return f"CountingCuckooRAM with {str(self.bloom_filter)}"

    def __repr__(self) -> str:
        return str(self)


class HeavyHittersRAM(RAM):
    def __init__(
        self,
        num_hitters=100,
        width=1000,
        depth=5,
        confidence=None,
        error_rate=None
    ):
        # print("Builiding HeavyHittersRAM...")
        self.bloom_filter = HeavyHitters(
            num_hitters=num_hitters,
            width=width,
            depth=depth,
            confidence=confidence,
            error_rate=error_rate
        )

    def encode_key(self, key: np.ndarray):
        return str().join(str(k * 1) for k in key)

    def add_member(self, key, inc_val: int = 1):
        key = self.encode_key(key)
        self.bloom_filter.add(key)

    def __contains__(self, key):
        key = self.encode_key(key)
        return self.bloom_filter.check(key) > 0

    def __getitem__(self, key):
        key = self.encode_key(key)
        return self.bloom_filter.check(key)

    def __str__(self) -> str:
        return f"HeavyHitters with {str(self.bloom_filter)}"

    def __repr__(self) -> str:
        return str(self)

class StreamThresholdRAM(RAM):
    def __init__(
        self,
        threshold=100,
        width=1000,
        depth=5,
        confidence=None,
        error_rate=None
    ):
        # print("Builiding HeavyHittersRAM...")
        self.bloom_filter = StreamThreshold(
            threshold=threshold,
            width=width,
            depth=depth,
            confidence=confidence,
            error_rate=error_rate
        )

    def encode_key(self, key: np.ndarray):
        return str().join(str(k * 1) for k in key)

    def add_member(self, key, inc_val: int = 1):
        key = self.encode_key(key)
        self.bloom_filter.add(key)

    def __contains__(self, key):
        key = self.encode_key(key)
        return self.bloom_filter.check(key) > 0

    def __getitem__(self, key):
        key = self.encode_key(key)
        return self.bloom_filter.check(key)

    def __str__(self) -> str:
        return f"StreamThreshold with {str(self.bloom_filter)}"

    def __repr__(self) -> str:
        return str(self)
