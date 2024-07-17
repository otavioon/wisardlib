import tempfile
from pathlib import Path

import numpy as np
import numba.core.types
from numba.typed import Dict
from typing import Hashable
from probables import (
    CountingBloomFilter,
    CountMinSketch,
    CountMeanSketch,
    CountMeanMinSketch,
    HeavyHitters,
    StreamThreshold,
    CountingCuckooFilter,
)

from .base import RAM, JoinableRAM
from wisardlib.config.type_definitions import BooleanArray


def size_of_bloom(filter) -> int:
    with tempfile.NamedTemporaryFile() as tmp:
        filter.export(tmp.name)
        tmp.flush()
        return Path(tmp.name).stat().st_size


class CountingBloomFilterRAM(JoinableRAM):
    def __init__(
        self, est_elements: int = 1000, false_positive_rate: float = 0.05
    ):
        self.bloom_filter = CountingBloomFilter(
            est_elements=est_elements, false_positive_rate=false_positive_rate
        )

    # def encode_key(self, key: BooleanArray):
    #     return str().join(str(k * 1) for k in key)

    def add_member(self, key: BooleanArray, inc_val: int = 1):
        key = self.encode_key(key)
        self.bloom_filter.add(key)

    def join(self, other: "CountingBloomFilterRAM"):
        self.bloom_filter = self.bloom_filter.union(other.bloom_filter)

    def __contains__(self, key: BooleanArray):
        key = self.encode_key(key)
        return self.bloom_filter.check(key) > 0

    def __getitem__(self, key: BooleanArray):
        key = self.encode_key(key)
        return self.bloom_filter.check(key)

    def __str__(self) -> str:
        return f"CountingBloomFilterRAM with {str(self.bloom_filter)}"

    def __repr__(self) -> str:
        return str(self)

    def size(self) -> int:
        return size_of_bloom(self.bloom_filter)
    
    def false_positive_rate(self) -> float:
        return self.bloom_filter.false_positive_rate


class _CMSRam(JoinableRAM):
    RAM_cls = None
    RAM_name = ""

    def __init__(
        self,
        width: int = None,
        depth: int = None,
        confidence: float = None,
        soft_error_rate: float = None,
    ):
        self.bloom_filter = self.RAM_cls(
            width=width,
            depth=depth,
            confidence=confidence,
            error_rate=soft_error_rate,
        )

    # def encode_key(self, key: BooleanArray):
    #     return str().join(str(k * 1) for k in key)

    def add_member(self, key: BooleanArray, inc_val: int = 1):
        key = self.encode_key(key)
        self.bloom_filter.add(key)

    def join(self, other):
        self.bloom_filter.join(other.bloom_filter)

    def __contains__(self, key: BooleanArray):
        key = self.encode_key(key)
        return self.bloom_filter.check(key) > 0

    def __getitem__(self, key: BooleanArray):
        key = self.encode_key(key)
        return self.bloom_filter.check(key)

    def __str__(self) -> str:
        return f"{self.RAM_name} with {str(self.bloom_filter)}"

    def __repr__(self) -> str:
        return str(self)

    def size(self) -> int:
        return size_of_bloom(self.bloom_filter)
    
    def false_positive_rate(self) -> float:
        return self.bloom_filter.error_rate


class CountMinSketchRAM(_CMSRam):
    RAM_cls = CountMinSketch
    RAM_name = "CountMinSketchRAM"


class CountMeanSketchRAM(_CMSRam):
    RAM_cls = CountMeanSketch
    RAM_name = "CountMeanSketchRAM"


class CountMeanMinSketchRAM(_CMSRam):
    RAM_cls = CountMeanMinSketch
    RAM_name = "CountMeanMinSketchRAM"


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

    # def encode_key(self, key: BooleanArray):
    #     return str().join(str(k * 1) for k in key)

    def add_member(self, key: BooleanArray, inc_val: int = 1):
        key = self.encode_key(key)
        self.bloom_filter.add(key)

    def __contains__(self, key: BooleanArray):
        key = self.encode_key(key)
        return self.bloom_filter.check(key) > 0

    def __getitem__(self, key: BooleanArray):
        key = self.encode_key(key)
        return self.bloom_filter.check(key)

    def __str__(self) -> str:
        return f"CountingCuckooRAM with {str(self.bloom_filter)}"

    def __repr__(self) -> str:
        return str(self)

    def size(self) -> int:
        return size_of_bloom(self.bloom_filter)
    
    def false_positive_rate(self) -> float:
        return self.bloom_filter.error_rate


class HeavyHittersRAM(JoinableRAM):
    def __init__(
        self,
        num_hitters=100,
        width=1000,
        depth=5,
        confidence=None,
        error_rate=None,
    ):
        # print("Builiding HeavyHittersRAM...")
        self.bloom_filter = HeavyHitters(
            num_hitters=num_hitters,
            width=width,
            depth=depth,
            confidence=confidence,
            error_rate=error_rate,
        )

    # def encode_key(self, key: BooleanArray):
    #     return str().join(str(k * 1) for k in key)

    def add_member(self, key: BooleanArray, inc_val: int = 1):
        key = self.encode_key(key)
        self.bloom_filter.add(key)

    def join(self, other: "HeavyHittersRAM"):
        self.bloom_filter.join(other.bloom_filter)

    def __contains__(self, key: BooleanArray):
        key = self.encode_key(key)
        return self.bloom_filter.check(key) > 0

    def __getitem__(self, key: BooleanArray):
        key = self.encode_key(key)
        return self.bloom_filter.check(key)

    def __str__(self) -> str:
        return f"HeavyHitters with {str(self.bloom_filter)}"

    def __repr__(self) -> str:
        return str(self)

    def size(self) -> int:
        return size_of_bloom(self.bloom_filter)

    def false_positive_rate(self) -> float:
        return self.bloom_filter.error_rate

class StreamThresholdRAM(JoinableRAM):
    def __init__(
        self,
        threshold=100,
        width=1000,
        depth=5,
        confidence=None,
        error_rate=None,
    ):
        # print("Builiding HeavyHittersRAM...")
        self.bloom_filter = StreamThreshold(
            threshold=threshold,
            width=width,
            depth=depth,
            confidence=confidence,
            error_rate=error_rate,
        )

    # def encode_key(self, key: BooleanArray):
    #     return str().join(str(k * 1) for k in key)

    def add_member(self, key: BooleanArray, inc_val: int = 1):
        key = self.encode_key(key)
        self.bloom_filter.add(key)

    def join(self, other: "StreamThresholdRAM"):
        self.bloom_filter.join(other.bloom_filter)

    def __contains__(self, key: BooleanArray):
        key = self.encode_key(key)
        return self.bloom_filter.check(key) > 0

    def __getitem__(self, key: BooleanArray):
        key = self.encode_key(key)
        return self.bloom_filter.check(key)

    def __str__(self) -> str:
        return f"StreamThreshold with {str(self.bloom_filter)}"

    def __repr__(self) -> str:
        return str(self)

    def size(self) -> int:
        return size_of_bloom(self.bloom_filter)
    
    def false_positive_rate(self) -> float:
        return self.bloom_filter.error_rate
