import numpy as np

from wisardlib.rams.dict_ram import DictRAM
from wisardlib.wisard import Discriminator, WiSARD
from wisardlib.utils import untie_by_first_class

# from wisardlib.hasher import BinaryHasher


def test_dict_ram():
    ram = DictRAM()
    samples = np.array(
        [
            [False, False, True],
            [False, False, True],
            [True, True, True],
        ],
        dtype=bool,
    )
    for sample in samples:
        ram.add_member(sample)

    assert ram[samples[0]] == 2
    assert samples[0] in ram
    assert ram[np.array([True])] == 0
    assert np.array([True]) not in ram


def test_dict_ram_discriminator():
    rams = [DictRAM(), DictRAM()]
    d = Discriminator(rams)
    assert d[0] == rams[0]
    assert d[1] == rams[1]


def test_dict_ram_discriminator_fit_predict():
    d = Discriminator([DictRAM(), DictRAM()])
    sample_2d = [np.array([False, False, False]), np.array([True, True, True])]

    d.fit(sample_2d)

    results = d.predict(sample_2d)
    assert results == 2

    sample_2d = [np.array([False, False, False]), np.array([False, False, False])]
    results = d.predict(sample_2d)
    assert results == 1

    sample_2d = [np.array([True, True, True]), np.array([False, False, False])]
    results = d.predict(sample_2d)
    assert results == 0


def test_dict_ram_discriminator_bleach():
    d = Discriminator([DictRAM(), DictRAM()])
    samples_2d = np.array(
        [
            [
                # Sample 0
                [0, 0, 0],  # RAM 0
                [1, 1, 1],  # RAM 1
            ],
            [
                # Sample 1
                [0, 0, 0],  # RAM 0
                [1, 1, 1],  # RAM 1
            ],
            [
                # Sample 2
                [0, 0, 0],  # RAM 0
                [0, 1, 1],  # RAM 1
            ],
        ],
        dtype=bool,
    )

    for sample in samples_2d:
        d.fit(sample)

    # Sample with 1 errored
    test_sample = np.array(
        [
            # Sample 0
            [0, 0, 0],  # RAM 0
            [0, 0, 0],  # RAM 1
        ],
        dtype=bool,
    )

    d.bleach = 1
    results = d.predict(test_sample)
    assert results == 1

    d.bleach = 3
    results = d.predict(test_sample)
    assert results == 1

    d.bleach = 4
    results = d.predict(test_sample)
    assert results == 0

    # Sample with 0 errored
    test_sample = np.array(
        [
            # Sample 0
            [0, 0, 0],  # RAM 0
            [1, 1, 1],  # RAM 1
        ],
        dtype=bool,
    )

    d.bleach = 1
    results = d.predict(test_sample)
    assert results == 2

    d.bleach = 3
    results = d.predict(test_sample)
    assert results == 1

    d.bleach = 4
    results = d.predict(test_sample)
    assert results == 0

    # Sample with 0 errored
    test_sample = np.array(
        [
            # Sample 0
            [0, 0, 0],  # RAM 0
            [1, 1, 1],  # RAM 1
        ],
        dtype=bool,
    )

    # Per RAM bleach
    d.bleach = [1, 3]
    results = d.predict(test_sample)
    assert results == 1

    d.bleach = [3, 3]
    results = d.predict(test_sample)
    assert results == 1

    d.bleach = [3, 1]
    results = d.predict(test_sample)
    assert results == 2

    d.bleach = [50, 41]
    results = d.predict(test_sample)
    assert results == 0


def test_wisard():
    X_train = np.array(
        [
            [0, 0, 0, 0],  # Sample 0
            [0, 0, 0, 1],  # Sample 1
            [0, 0, 0, 1],  # Sample 2
            [0, 0, 1, 1],  # Sample 3
            [1, 1, 1, 1],  # Sample 4
            [1, 1, 1, 1],  # Sample 5
            [0, 1, 1, 1],  # Sample 6
            [0, 1, 1, 1],  # Sample 6
        ],
        dtype=bool,
    )

    y_train = [0, 0, 0, 0, 1, 1, 1, 1]

    indices = [0, 1, 2, 3]
    tuple_size = 2  # indices[0:1] from sample to discriminator 0, and
    # indices[2:3] from sample to discriminator 1 (if not suffled)
    discriminators = [
        Discriminator(rams=[DictRAM(), DictRAM()]),
        Discriminator(rams=[DictRAM(), DictRAM()]),
    ]

    # Create the model
    model = WiSARD(
        discriminators=discriminators,
        indices=indices,
        tuple_size=tuple_size,
        use_tqdm=False,
    )
    model.fit(X_train, y_train)

    # Let's predict
    X_test = np.array([[0, 0, 0, 1], [1, 1, 1, 1]], dtype=bool)

    y_test = [0, 1]

    result = model.predict(X_test)
    result, ties = untie_by_first_class(result)

    assert ties == 0
    np.testing.assert_array_equal(result, np.asarray(y_test))


#
#
#
#     w = WiSARD(discriminators=[Discriminator([DictRAM(), DictRAM()]), Discriminator([DictRAM(), DictRAM()])], indices=list(range(4)), tuple_size=2, shuffle_indices=True)
# w.bleach = 1
#
# X_train = np.array([
#     [0, 0, 0, 0],
#     [0, 0, 0, 1],
#     [0, 0, 1, 1],
#     [0, 0, 0, 0],
#     [1, 1, 0, 0],
#     [1, 1, 0, 1],
#     [1, 1, 0, 0],
#     [1, 0, 0, 0],
# ], dtype=bool)
#
# y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])
#
#
# w.fit(X_train, y_train)
#
#
# X_test = np.array([
#     [0, 0, 0, 0],
#     [0, 0, 0, 1],
#     [1, 0, 0, 0],
#     [1, 1, 0, 0],
# ], dtype=bool)
#
# y_test = np.array([0, 0, 1, 1])
#
# y_pred = w.predict(X_test)
#
# y_pred, ties = untie_by_first_class(y_pred)
#
# print(f"Real: {y_test}")
# print(f"Predicted: {y_pred}")
# print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
# print(f"Number of ties: {ties}")
