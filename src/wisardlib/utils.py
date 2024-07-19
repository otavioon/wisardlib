from typing import Tuple
import numpy as np
import tqdm


# def untie_by_first_class(
#     y_pred: np.ndarray, use_tqdm: bool = True
# ) -> Tuple[np.ndarray, int]:
#     if use_tqdm:
#         y_pred = tqdm.tqdm(y_pred, desc="Untying...")
#     final_pred = []
#     ties = 0

#     for pred in y_pred:
#         best_classes = np.where(pred == pred.max())[0]
#         if len(best_classes) > 1:
#             ties += 1
#         # Always pick the first class in case of doubt...
#         final_pred.append(best_classes[0])

#     return np.array(final_pred), ties
