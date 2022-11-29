import numpy as np
from wisardlib.encoders.thermometer import ThermometerEncoder

def test_thermometer_encoding():
    buckets = [25.0, 50.0, 75.0]

    X = np.array([0, 1, 25, 26, 50, 78, 100])
    expected_X = np.array([
        [False, False, False],
        [False, False, False],
        [False, False, True],
        [False, False, True],
        [False, True,  True],
        [True,  True,  True],
        [True,  True,  True]
    ], dtype=bool)

    encoder = ThermometerEncoder(3)
    encoder.fit(X)
    assert len(encoder.buckets) == len(buckets)
    assert all([a == b for a, b in zip(encoder.buckets, buckets)])

    X_encoded = encoder.transform(X)

    assert np.array_equal(expected_X, X_encoded)
