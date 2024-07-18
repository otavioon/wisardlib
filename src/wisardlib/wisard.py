from collections import defaultdict
import random
import itertools

from typing import Dict, List, Union, Optional, Iterable
from numbers import Real

import numpy as np
import tqdm

from wisardlib.config.type_definitions import ByteArray, Address, Sample
from wisardlib.rams.base import RAM
from wisardlib.bleaching.base import Bleaching, HighestResponseBleaching


class Discriminator:
    def __init__(
        self,
        rams: List[RAM],
    ):
        """A set of Random Access Memories (RAMs) that will be used to discriminate
        a class.

        Parameters
        ----------
        rams : List[RAM]
            List of Random Access Memories
        """
        self._rams: List[RAM] = rams

    def fit(self, X: List[Address], y=None) -> "Discriminator":
        """Fit model based on the input. The input is a single sample with
        multiple addresses. Each address will be stored in a respective RAM.

        Parameters
        ----------
        X : List[Address]
            List of addresses. Each address will be addressed to each RAM.
        y : type
            Not used. Each discriminator is used for a single class.

        Returns
        -------
        Discriminator
            The self object.

        Raises
        ------
        ValueError
            If the number of RAMs is different from the number of input
            addresses.

        """
        # Do checking
        if len(X) != len(self._rams):
            raise ValueError("X must have same length as number of RAMs")
        
        # Add members to the respective RAMs
        for ram, sample in zip(self._rams, X):
            ram.add_member(sample)
        
        return self

    def predict(self, X: List[Address]) -> List[int]:
        """Make a prediction based on the input. The input is a single sample
        with multiple addresses. Each address will be checked against each RAM.
        The response of each RAM will be returned

        Parameters
        ----------
        X : List[Address]
            List of addresses. Each address will be checked against each RAM.

        Returns
        -------
        List[int]
            A list of responses (frequency) of each RAM.

        Raises
        ------
        ValueError
            If the number of RAMs is different from the number of input
            addresses.

        """
        # Do checking
        if len(X) != len(self._rams):
            raise ValueError("X must have same length as number of RAMs")

        return [ram[x] for ram, x in zip(self._rams, X)]

    def join(self, other: "Discriminator"):
        for r, other_r in zip(self._rams, other._rams):
            r.join(other_r)

    def __getitem__(self, key: int) -> RAM:
        """Get a single RAM with python's subscribed operator.

        Parameters
        ----------
        key : int
            The index of the RAM.

        Returns
        -------
        RAM
            A single RAM.

        """
        return self._rams[key]

    def __len__(self) -> int:
        """Number of RAMs in the discriminator.

        Returns
        -------
        int
            The number of RAMs.
        """
        return len(self._rams)

    def __str__(self) -> str:
        return f"Discriminator with {len(self._rams)} RAMS"

    def __repr__(self) -> str:
        return str(self)

    def size(self) -> int:
        """The size of the discriminator (sum of size of all rams), in bytes.

        Returns
        -------
        int
            The size of the discriminator.
        """
        return sum(r.size() for r in self._rams)

    def false_positive_rate(self) -> float:
        return sum(r.false_positive_rate() for r in self._rams) / len(self._rams)


class WiSARD:
    def __init__(
        self,
        discriminators: (
            Discriminator | List[Discriminator] | Dict[int, Discriminator]
        ),
        tuple_size: int,
        bleaching_method: Optional[Bleaching] = None,
        use_tqdm: bool = True,
        mapping: Optional[List[int]] = None,
    ):
        if isinstance(discriminators, dict):
            self.discriminators = discriminators
        elif isinstance(discriminators, Iterable):
            self.discriminators = {i: d for i, d in enumerate(discriminators)}
        elif isinstance(discriminators, Discriminator):
            self.discriminators = {0: discriminators}
        else:
            raise TypeError(
                "Invalid type for discriminators. It must be a list, a dict or a single Discriminator object"
            )

        self.num_rams_per_discriminator = len(
            list(self.discriminators.values())[0]
        )
        assert all(
            len(d) == self.num_rams_per_discriminator
            for d in self.discriminators.values()
        ), "All discriminators must have the same number of RAMs"

        self.tuple_size = tuple_size
        self.bleaching_method = bleaching_method or HighestResponseBleaching()
        self.use_tqdm = use_tqdm
        self.min_input_size = self.num_rams_per_discriminator * self.tuple_size
        self.mapping = mapping
        if mapping is None:
            self.mapping = list(range(self.min_input_size))
            random.shuffle(self.mapping)

    def __getitem__(self, key):
        """Get the discriminator using the subscribed operator."""
        return self.discriminators[key]

    def _addressify(self, x: Sample) -> List[Address]:
        """Divide the input in a list of sub-samples, each one with the size of
        the tuple_size. The mapping is used to select the indices of the input
        that will be used.

        Parameters
        ----------
        x : Sample
            The input to be divided.

        Returns
        -------
        List[Address]
            A list of sub-samples (with random mapping) with the size of
            tuple_size.

        Raises
        ------
        ValueError
            If the input size is less than the minimum input size.
        """
        if len(x) < self.min_input_size:
            raise ValueError(
                f"Input size must be at least {self.min_input_size} but got {len(x)}"
            )
        # Reindex the array
        x = x[self.mapping]
        return [
            x[i : i + self.tuple_size]
            for i in range(0, len(x), self.tuple_size)
        ]

    def fit(self, X: Iterable[Sample], y: np.ndarray) -> "WiSARD":
        """Fit the model over given input samples.

        Parameters
        ----------
        X : Iterable[Sample]
            A list of samples. Each sample will be divided in sub-samples
            and each sub-sample will be stored in a respective RAM.
        y : np.ndarray
            The target classes for each sample. The class will be used to
            select the discriminator.

        Returns
        -------
        WiSARD
            The self object.

        """
        it = sorted(zip(X, y), key=lambda x: x[1]) 

        # If use tqdm, create an tqdm iterator at each sample
        if self.use_tqdm:
            it = tqdm.tqdm(it, total=len(X), leave=True, position=0, desc="Fit")
        # Iterate over inputs
        for _X, _y in it:
            # Transform each input as a list of subsamples (based on indices)
            addresses = self._addressify(X[i])
            self.discriminators[y[i]].fit(addresses)
        return self

    def predict_proba(self, X: Iterable[Sample]) -> List[Dict[int, List[int]]]:
        per_sample_responses = []

        it = range(len(X))
        # If use tqdm, create an tqdm iterator at each sample
        if self.use_tqdm:
            it = tqdm.tqdm(
                it, total=len(X), leave=True, position=0, desc="Predict"
            )

        for i in it:
            responses = dict()
            x = X[i]
            addresses = self._addressify(x)
            for disc_name, disc in self.discriminators.items():
                responses[disc_name] = disc.predict(addresses)
            per_sample_responses.append(responses)

        return per_sample_responses

    def predict(self, X: Iterable[Sample]) -> np.ndarray:
        """Predict the class of each sample.

        Parameters
        ----------
        X : Iterable[Sample]
            A list of samples.

        Returns
        -------
        np.ndarray
            The predicted class for each sample.
        """
        per_sample_responses = []

        it = range(len(X))
        # If use tqdm, create an tqdm iterator at each sample
        if self.use_tqdm:
            it = tqdm.tqdm(
                it, total=len(X), leave=True, position=0, desc="Predict"
            )

        for i in it:
            responses = dict()
            x = X[i]
            addresses = self._addressify(x)
            for disc_name, disc in self.discriminators.items():
                responses[disc_name] = disc.predict(addresses)
            # Apply bleaching to get the final prediction
            pred = self.bleaching_method(responses)
            per_sample_responses.append(pred)

        return np.array(per_sample_responses)

    def join(self, other: "WiSARD"):
        for d, other_d in zip(self.discriminators, other.discriminators):
            d.join(other_d)

    def __len__(self) -> int:
        """Number of discriminators in the WiSARD.

        Returns
        -------
        int
            The number of discriminators.
        """
        return len(self.discriminators)

    def __str__(self) -> str:
        return f"WiSARD with {len(self.discriminators)} discriminators ({self.num_rams_per_discriminator} RAMs each)"

    def __repr__(self) -> str:
        return str(self)

    def size(self) -> int:
        """The size of the WiSARD (sum of size of all discriminators), in bytes.

        Returns
        -------
        int
            The size of the WiSARD, in bytes.
        """
        return sum(d.size() for d in self.discriminators.values())
