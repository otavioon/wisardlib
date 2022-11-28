from abc import abstractmethod

from typing import Hashable


class RAM:
    """Random access memory implementation."""

    @abstractmethod
    def add_member(self, key: Hashable, inc_val: int = 1):
        raise NotImplementedError

    @abstractmethod
    def __contains__(self, key: Hashable) -> bool:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, key: Hashable) -> int:
        raise NotImplementedError
