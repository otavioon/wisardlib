from wisardlib.encoders.thermometer import ThermometerEncoder
from pathlib import Path
import numpy as np
import random

from wisardlib.rams.dict_ram import DictRAM
from wisardlib.rams.bloom_filter_ram import HeavyHittersRAM
from wisardlib.builder import build_symmetric_wisard
from wisardlib.utils import untie_by_first_class
from sklearn.metrics import accuracy_score


data_path = Path("data/mnist.pkl")
(x_train, y_train), (x_test, y_test) = np.load(data_path, allow_pickle=True)

sub_sample_indices_train = random.sample(range(len(x_train)), 1000)
sub_sample_indices_test = random.sample(range(len(x_test)), 1000)

x_train = x_train[sub_sample_indices_train]
y_train = y_train[sub_sample_indices_train]
x_test = x_test[sub_sample_indices_test]
y_test = y_test[sub_sample_indices_test]


encoder = ThermometerEncoder(16)
encoder.fit(x_train)
print("Encoding training set...")
x_train = [x.ravel() for x in encoder.transform(x_train)]
print("Encoding test set...")
x_test  = [x.ravel() for x in encoder.transform(x_test)]
indices = list(range(len(x_train[0].ravel())))
tuple_size = 16

print("Creating WiSARD...")
w = build_symmetric_wisard(
    RAM_cls=HeavyHittersRAM,
    RAM_creation_kwargs=None,
    number_of_rams_per_discriminator=len(indices)//tuple_size,
    number_of_discriminators=10,
    indices=indices,
    tuple_size=tuple_size,
    shuffle_indices=True
)

print("Fitting model...")
w.fit(x_train, y_train)

print("Predicting...")
y_pred = w.predict(x_test)

y_pred, ties = untie_by_first_class(y_pred)

print(f"Real: {y_test}")
print(f"Predicted: {y_pred}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Number of ties: {ties}")
