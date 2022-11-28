import random
import itertools

from typing import List, Union, Optional, Iterable
from numbers import Real

import numpy as np
import tqdm
from wisardlib.rams.base import RAM


class Discriminator:
    def __init__(
        self,
        rams: List[RAM],
        bleach: Real | List[Real] = 1,
        count_responses: bool = True,
    ):
        self._rams: List[RAM] = rams
        self._bleach = None
        self._count_responses = count_responses
        self.bleach = bleach

    def fit(self, X: List[np.ndarray], y=None):
        if len(X) != len(self._rams):
            raise ValueError("X must have same length as number of RAMs")
        for ram, sample in zip(self._rams, X):
            ram.add_member(sample)
        return self

    def predict(self, X: List[np.ndarray]):
        if len(X) != len(self._rams):
            raise ValueError("X must have same length as number of RAMs")

        if isinstance(self._bleach, Real):
            bleach_vals = itertools.repeat(self._bleach, times=len(X))
        else:
            bleach_vals = self._bleach

        if self._count_responses:
            return sum(
                int(ram[x] >= bleach)
                for ram, bleach, x in zip(self._rams, bleach_vals, X)
            )
        else:
            return sum(
                0 if ram[x] < bleach else ram[x]
                for ram, bleach, x in zip(self._rams, bleach_vals, X)
            )

        for ram, bleach, x in zip(self._rams, bleach_vals, X):
            value = ram[x]
            if self._count_responses:
                response += int(value >= bleach)
            else:
                response += 0 if value < bleach else value
        return response

    @property
    def bleach(self):
        return self._bleach

    @bleach.setter
    def bleach(self, value: Real | List[Real]):
        if isinstance(value, list):
            if len(value) != len(self._rams):
                raise ValueError("Length of bleach must be the same as number of RAMS")
            self._bleach = value
        elif isinstance(value, Real):
            self._bleach = value
        else:
            raise TypeError("Bleach must be list or real number")

    def __getitem__(self, key) -> RAM:
        return self._rams[key]

    def __str__(self) -> str:
        return f"Discriminator with {len(self._rams)} RAMS"

    def __repr__(self) -> str:
        return str(self)

    def __len__(self) -> int:
        return len(self._rams)


class WiSARD:
    def __init__(
        self,
        discriminators: List[Discriminator],
        indices: List[int],
        tuple_size: int | List[int] | List[slice],
        shuffle_indices: bool = False,
        use_tqdm: bool = True,
    ):
        assert isinstance(discriminators, Iterable), "Must be a list of discriminators"
        assert isinstance(indices, Iterable), "Must be a list of indices"

        self._discriminators: List[Discriminator] = discriminators
        _indices = indices.copy()
        if shuffle_indices:
            random.shuffle(_indices)
        self._indices: List[np.ndarray] = self._calculate_indices(_indices, tuple_size)

        for d in self._discriminators:
            assert len(d) >= len(
                self._indices
            ), "Number of RAMS per discriminator must be greater or equal the number of addresses"

        self.use_tqdm = use_tqdm

    @staticmethod
    def _calculate_indices(
        indices, tuple_size: int | List[int] | List[List[int]]
    ) -> List[slice]:
        if isinstance(tuple_size, list):
            if isinstance(tuple_size[0], list):
                # Allows overlap by specifying slices manually
                _indices = np.array(indices)
                return [_indices[s] for s in tuple_size]

        if isinstance(tuple_size, int):
            tuple_size = [tuple_size]

        start = 0
        must_stop = False
        slices = []
        while not must_stop:
            for size in tuple_size:
                if start >= len(indices):
                    must_stop = True
                    break
                slices.append(slice(start, start + size, 1))
                start += size
        return [indices[s] for s in slices]

    def __getitem__(self, key):
        return self._discriminators[key]

    @property
    def bleach(self) -> List[List[Real]] | List[Real]:
        return [d.bleach for d in self._discriminators]

    @bleach.setter
    def bleach(self, value: int | List[int]):
        for d in self._discriminators:
            d.bleach = value

    @property
    def indices(self):
        return self._indices

    @property
    def min_val(self) -> int:
        return min(d.min_val for d in self._discriminators)

    @property
    def max_val(self) -> int:
        return max(d.max_val for d in self._discriminators)

    def _reindex_sample(self, x: np.ndarray) -> List[np.ndarray]:
        return [x[i] for i in self.indices]

    def fit(self, X: np.ndarray, y: np.ndarray):
        it = range(len(X))
        if self.use_tqdm:
            it = tqdm.tqdm(
                it, total=len(X), leave=True, position=0, desc="Fitting model..."
            )
        for i in it:
            sample = self._reindex_sample(X[i])
            self._discriminators[y[i]].fit(sample)
        return self

    def predict(self, X: np.ndarray):
        it = range(len(X))
        if self.use_tqdm:
            it = tqdm.tqdm(
                it, total=len(X), leave=True, position=0, desc="Predicting   ..."
            )
        y_pred = []
        for i in it:
            sample = self._reindex_sample(X[i])
            responses = np.array([d.predict(sample) for d in self._discriminators])
            y_pred.append(responses)  # np.where(responses == responses.max())[0])
        return np.array(y_pred)

    #
    # def _reindex_sample(self, x: np.ndarray) -> List[np.ndarray]:
    #     xs = x[self._indices]
    #     xs = [np.pad(x[s], (0, (x[s].size - (s.stop - s.start)))) for s in self._slices]
    #     return xs
    #
    # def _fit_sample(self, x: np.ndarray, discriminator: Discriminator):
    #     xs = self._reindex_sample(x)
    #     discriminator.fit(xs)
    #
    # def _predict_sample(self, x: np.ndarray, soft_error_rate: float = 0.0):
    #     xs = self._reindex_sample(x)
    #     responses = np.array(
    #         [
    #             d.predict(xs, soft_error_rate=soft_error_rate)
    #             for d in self._discriminators
    #         ]
    #     )
    #     max_response = responses.max()
    #     return np.where(responses == max_response)[0]
    #
    # def fit(self, X: np.ndarray, y: np.ndarray, use_tqdm: bool = False):
    #     it = range(len(X))
    #     if use_tqdm:
    #         it = tqdm.tqdm(
    #             it, total=len(X), leave=True, position=0, desc="Fitting model"
    #         )
    #     for i in it:
    #         self._fit_sample(X[i], self._discriminators[y[i]])
    #
    #     return self
    #
    # def predict(
    #     self,
    #     X: np.ndarray,
    #     y: np.ndarray,
    #     use_tqdm: bool = False,
    #     soft_error_rate: float = 0.0,
    #     bleach: Optional[Union[int, List[int]]] = None,
    # ):
    #     if bleach is not None:
    #         self.bleach = bleach
    #     y_pred = [self._predict_sample(x, soft_error_rate) for x in X]
    #     return y_pred

    def __str__(self) -> str:
        return f"WiSARD with {len(self._discriminators)} discriminator. Min, max: [{self.min_val}, {self.max_val}]"

    def __repr__(self) -> str:
        return str(self)


# def simple_discriminators_build(
#     num_rams: int,
#     ram_cls: type,
#     ram_kwargs: dict = None,
#     num_disc: int = 1,
#     bleach: int = 1,
# ) -> List[Discriminator]:
#     ram_kwargs = ram_kwargs or dict()
#     return [
#         Discriminator([ram_cls(**ram_kwargs) for j in range(num_rams)])
#         for i in range(num_disc)
#     ]
