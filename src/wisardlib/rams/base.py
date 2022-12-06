from abc import abstractmethod

from wisardlib.config.type_definitions import BooleanArray

class RAM:
    """Random access memory implementation."""

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
