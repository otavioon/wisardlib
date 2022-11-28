import numpy as np

from wisardlib.rams.bloom_filter_ram import CountMinSketchRAM
from wisardlib.builder import build_symmetric_wisard
from wisardlib.utils import untie_by_first_class
from sklearn.metrics import accuracy_score
# from wisardlib.hasher import BinaryHasher
w = build_symmetric_wisard(
    RAM_cls=CountMinSketchRAM,
    RAM_creation_kwargs=None,
    number_of_rams_per_discriminator=2,
    number_of_discriminators=2,
    indices=list(range(4)),
    tuple_size=2,
    shuffle_indices=True
)

#WiSARD(discriminators=[Discriminator([CountMinSketchRAM(), CountMinSketchRAM()]), Discriminator([CountMinSketchRAM(), CountMinSketchRAM()])], indices=list(range(4)), tuple_size=2, shuffle_indices=True)
w.bleach = 1

X_train = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 1],
    [0, 0, 0, 0],
    [1, 1, 0, 0],
    [1, 1, 0, 1],
    [1, 1, 0, 0],
    [1, 0, 0, 0],
], dtype=bool)

y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])


w.fit(X_train, y_train)


X_test = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 0],
    [1, 1, 0, 0],
], dtype=bool)

y_test = np.array([0, 0, 1, 1])

y_pred = w.predict(X_test)

y_pred, ties = untie_by_first_class(y_pred)

print(f"Real: {y_test}")
print(f"Predicted: {y_pred}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Number of ties: {ties}")
