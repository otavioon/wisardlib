import random
import itertools

from typing import List, Union, Optional, Iterable
from numbers import Real

import numpy as np
import tqdm

from wisardlib.config.type_definitions import BooleanArray
from wisardlib.rams.base import RAM


class Discriminator:
    """A set of Random Access Memory for a single class.

    Parameters
    ----------
    rams : List[RAM]
        List of Random Access Memories
    bleach : Real | List[Real]
        The bleach value (the default is 1).
        - If the value is a Real number, the bleach will the same to all RAMs
        - If it is a list of Real, the bleach value will be applyied to each RAM.
    count_responses : bool
        if true, count responses higher than bleach, else, return the responses
        (the default is True).
    """

    def __init__(
        self,
        rams: List[RAM],
        bleach: Real | List[Real] = 1,
        count_responses: bool = False,
    ):
        self._rams: List[RAM] = rams
        self._bleach = None
        self._count_responses = count_responses
        self.bleach = bleach
        

    def fit(self, X: List[BooleanArray], y=None):
        """Fit model based on the input.

        Parameters
        ----------
        X : List[BooleanArray]
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
            If the number of RAMs is different from the number of input addresses.

        """
        # Do checking
        if len(X) != len(self._rams):
            raise ValueError("X must have same length as number of RAMs")
        
        # Add members to the respective RAMs
        for ram, sample in zip(self._rams, X):
            ram.add_member(sample)
        
        return self

    def predict(self, X: List[BooleanArray]) -> np.ndarray:
        """Make predictions for each sample.

        Parameters
        ----------
        X : List[BooleanArray]
            List of addresses. Each address will be checked against each RAM.

        Returns
        -------
        np.ndarray
            The array with the scores for each class. If `_count_responses` is
            True, it will be returned it will be returned the number of times
            that each input response is greater than bleach. Else, it will be
            returned the sum of responses (frequency of the address) that is
            higher than bleach.

        Raises
        ------
        ValueError
            If the number of RAMs is different from the number of input addresses.

        """
        # Do checking
        if len(X) != len(self._rams):
            raise ValueError("X must have same length as number of RAMs")

        # If bleach is a number, it will be the same for all RAMs
        if isinstance(self._bleach, Real):
            bleach_vals = itertools.repeat(self._bleach, times=len(X))
        # Else: one singular bleach value to one RAMs
        else:
            bleach_vals = self._bleach

        # If True, count the number of times each response if higher than
        # respective RAM's bleach
        if self._count_responses:
            return sum(
                int(ram[x] >= bleach)
                for ram, bleach, x in zip(self._rams, bleach_vals, X)
            )
        # Else, return the sum of responses of responses higher than respective
        # RAM's bleach
        else:
            return sum(
                0 if ram[x] < bleach else ram[x]
                for ram, bleach, x in zip(self._rams, bleach_vals, X)
            )

        # for ram, bleach, x in zip(self._rams, bleach_vals, X):
        #     value = ram[x]
        #     if self._count_responses:
        #         response += int(value >= bleach)
        #     else:
        #         response += 0 if value < bleach else value
        # return response

    @property
    def bleach(self):
        """Bleach getter.

        Returns
        -------
        Real | List[Real]
            The bleach value.

        """
        return self._bleach

    @bleach.setter
    def bleach(self, value: Real | List[Real]):
        """Bleach setter.

        Parameters
        ----------
        value : Real | List[Real]
            If bleach is Real, the same bleach value will be applyied to each
            RAM. Else, one specific bleach will be applyied to each RAM.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If a list of bleaches is passed with different number of RAMs.

        TypeError
            If the `value` passed is not a Real number nor a list

        """
        if isinstance(value, list):
            if len(value) != len(self._rams):
                raise ValueError("Length of bleach must be the same as number of RAMS")
            self._bleach = value
        elif isinstance(value, Real):
            self._bleach = value
        else:
            raise TypeError("Bleach must be a list of real numbers or real number")

    def join(self, other: "Discriminator"):
        for r, other_r in zip(self._rams, other._rams):
            r.join(other_r)

    def __getitem__(self, key) -> RAM:
        """Get a single RAM with python's subscribed operator.

        Parameters
        ----------
        key : int | slice
            The key to be get.

        Returns
        -------
        RAM
            The RAMs gathered.

        """
        return self._rams[key]

    def __str__(self) -> str:
        return f"Discriminator with {len(self._rams)} RAMS"

    def __repr__(self) -> str:
        return str(self)

    def __len__(self) -> int:
        return len(self._rams)

    def size(self) -> int:
        return sum(r.size() for r in self._rams)

    def false_positive_rate(self) -> float:
        return sum(r.false_positive_rate() for r in self._rams) / len(self._rams)


class WiSARD:
    """Short summary.

    Parameters
    ----------
    discriminators : List[Discriminator]
        A list of RAM-discriminators. One for each desired class.
    indices : int | List[int]
        The list of indices that will be selected by each tuple or an int
        that will generate a list of sequential indices.
    tuple_size : int | List[int] | List[slice]
        Size of each tuple. For each RAM, `tuple_size` indices will be
        selected from `indices`, consecutively.
        The type can be:
        - An non-negative integer: the tuples of same size will be consecutively
            be selected from `indices`. For instance, if `tuple_size` == 3, then
            for each RAM, the following indices will be selected:
            [0 .. `tuple_size`, `tuple_size+1`..2*`tuple_size`, ...]
        - An list of integer: the indices will of each element will be:
            0..sum(tuple_size[:1]), sum(tuple_size[:1])..sum(tuple_size[:2]), ...
        - An list of 2-element lists. Each element is the start and end of the
            range of the `indices`.
        **NOTE**: `data[selected_indices] == RAM address`
        **NOTE**: The sum of `tuple_size` (if a list) must be equals `indices` length.
    shuffle_indices : bool
        Shuffle indices before selecting composing an address (the default is False).
    use_tqdm : bool
        Use tqdm progress bar (the default is True).

    """

    def __init__(
        self,
        discriminators: List[Discriminator],
        indices: List[int],
        tuple_size: int | List[int] | List[slice],
        shuffle_indices: bool = False,
        use_tqdm: bool = True,
    ):
        assert isinstance(discriminators, Iterable), "Must be a list of discriminators"

        self._discriminators: List[Discriminator] = discriminators
        if isinstance(indices, Iterable):
            _indices = indices.copy()
        elif isinstance(indices, int):
            _indices = list(range(indices))
        else:
            raise TypeError("Indices must be a list or a interger value")

        if shuffle_indices:
            random.shuffle(_indices)
        self._indices: List[slice] = self._calculate_indices(_indices, tuple_size)

        for d in self._discriminators:
            assert len(d) >= len(
                self._indices
            ), "Number of RAMS per discriminator must be greater or equal the number of addresses"

        self.use_tqdm = use_tqdm

    @staticmethod
    def _calculate_indices(
        indices, tuple_size: int | List[int] | List[slice]
    ) -> List[slice]:
        """Calculate the slice of `indices` that will be use for input to be
        used in each RAM.

        Parameters
        ----------
        indices : List[int]
            List of indices.
        tuple_size : int | List[int] | List[slice]
            The `tuple_size` or slices.

        Returns
        -------
        List[slice]
            List of slices to be used in the input data to get an address.

        """
        if isinstance(tuple_size, list):
            if isinstance(tuple_size[0], slice):
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
        """Get the discriminator using the subscribed operator."""
        return self._discriminators[key]

    @property
    def bleach(self) -> List[List[Real]] | List[Real]:
        """Get the bleach of each discriminator.

        Returns
        -------
        List[List[Real]] | List[Real]
            A list of bleach defined for each discriminator.

        """
        return [d.bleach for d in self._discriminators]

    @bleach.setter
    def bleach(self, value: Real | List[Real]):
        """Set the same bleach value to all discriminators.

        Parameters
        ----------
        value : Real | List[Real]
            The bleach value.

        Returns
        -------
        None

        """
        for d in self._discriminators:
            d.bleach = value

    @property
    def indices(self):
        """Get calculated indices."""
        return self._indices

    def _reindex_sample(self, x: BooleanArray) -> List[BooleanArray]:
        """Given a numpy array as input, create a list of subsamples acessing
        the input at each indices.

        Parameters
        ----------
        x : np.ndarray
            The input.

        Returns
        -------
        List[np.ndarray]
            List of sub-arrays, each for respective subset of indices.

        Raises
        ------
        ExceptionName
            Why the exception is raised.

        """
        return [x[i] for i in self.indices]

    def fit(self, X: BooleanArray, y: np.ndarray):
        """Fit the model over given input samples.

        Parameters
        ----------
        X : BooleanArray
            An array of samples.
        y : np.ndarray
            Each respective sample class (it will be indexed as the discriminator)

        Returns
        -------
        WiSARD
            The self object.

        """
        it = sorted(zip(X, y), key=lambda x: x[1]) 

        # If use tqdm, create an tqdm iterator at each sample
        if self.use_tqdm:
            it = tqdm.tqdm(
                it, total=len(X), leave=True, position=0, desc="Fitting model..."
            )
        # Iterate over inputs
        for _X, _y in it:
            # Transform each input as a list of subsamples (based on indices)
            sample = self._reindex_sample(_X)
            self._discriminators[_y].fit(sample)
        return self

    def predict(self, X: BooleanArray) -> np.ndarray:
        """Predict the score of each sample per discriminator.

        Parameters
        ----------
        X : BooleanArray
            An array of samples.

        Returns
        -----------
        np.ndarray (an matrix)
            A matrix where each row has the scores (responses from discriminators)
            for each sample of X. Each row is a vector where each column is the
            score (response) given by a discriminator.

        """
        it = range(len(X))
        # Iterate over each sample from X
        if self.use_tqdm:
            it = tqdm.tqdm(
                it, total=len(X), leave=True, position=0, desc="Predicting   ..."
            )
        # List with the responses of each disriminator per sample
        y_pred: List[np.ndarray] = []
        # Iterate over samples
        for i in it:
            # Reindex the input according to indices
            sample = self._reindex_sample(X[i])
            # Calculate the response for each discriminator
            responses = np.array([d.predict(sample) for d in self._discriminators])
            y_pred.append(responses)  # np.where(responses == responses.max())[0])
        return np.array(y_pred)

    def join(self, other: "WiSARD"):
        for d, other_d in zip(self._discriminators, other._discriminators):
            d.join(other_d)

    def __len__(self) -> int:
        return len(self._discriminators)

    def __str__(self) -> str:
        return f"WiSARD with {len(self._discriminators)} discriminators."

    def __repr__(self) -> str:
        return str(self)

    def size(self) -> int:
        return sum(d.size() for d in self._discriminators)

    def mean_false_positive_rate(self) -> float:
        return sum(d.false_positive_rate() for d in self._discriminators) / len(self._discriminators)