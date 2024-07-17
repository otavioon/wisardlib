from typing import Dict, List, Literal
import numpy as np


class Bleaching:
    def __init__(self, threshold: int = 1):
        self._threshold = threshold
        self._num_ties = 0

    @property
    def ties(self):
        return self._num_ties

    @property
    def threshold(self):
        return self._threshold

    def __call__(self, responses: Dict[int, List[int]]) -> int:
        if not responses:
            raise ValueError("No responses to evaluate")

        results = {
            disc_name: np.sum((np.array(resps) >= self._threshold) * 1)
            for disc_name, resps in responses.items()
        }

        # argmax
        max_key = max(results, key=results.get)

        self._num_ties = (
            sum(1 for v in results.values() if v == results[max_key]) - 1
        )

        return max_key


class HighestResponseBleaching(Bleaching):
    def __init__(self):
        self._num_ties = 0

    @property
    def ties(self):
        return self._num_ties

    def __call__(self, responses: Dict[int, List[int]]) -> int:
        if not responses:
            raise ValueError("No responses to evaluate")

        results = {
            disc_name: sum(resps) for disc_name, resps in responses.items()
        }
        # argmax
        max_key = max(results, key=results.get)
        # Number of keys that have the same value as the max_key (ties)
        self._num_ties = (
            sum(1 for v in results.values() if v == results[max_key]) - 1
        )

        return max_key


class LinearBleaching(Bleaching):
    def __init__(self, step_size: int = 5):
        self._num_ties = 0
        self._threshold = 0
        self._step_size = step_size

    @property
    def ties(self):
        return self._num_ties

    @property
    def threshold(self):
        return self._threshold

    def __call__(self, responses: Dict[int, List[int]]) -> int:
        if not responses:
            raise ValueError("No responses to evaluate")

        biggest = max([max(resps) for resps in responses.values()])        
        min_ties = np.inf
        best_bleach = 1


        for b in range(1, biggest, self._step_size):
            # print(f"Trying threshold: {b}...")
            bleach = Bleaching(threshold=b)
            pred = bleach(responses)
            num_ties = bleach.ties
            
            if num_ties < min_ties:
                min_ties = num_ties
                best_bleach = b         

            if num_ties == 0:
                break

        # print(f"[BEST] Found threshold: {b}")
        # print(f"[BEST] Ties: {min_ties}")

        bleach = Bleaching(threshold=best_bleach)
        pred = bleach(responses)
        self._num_ties = bleach.ties
        self._threshold = best_bleach

        return pred
