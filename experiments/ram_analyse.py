from numba import jit
from pathlib import Path
import numpy as np
import pandas as pd
import random
import logging
import traceback
import argparse
import time
import json
from dataclasses import dataclass, field
from typing import List


from wisardlib.builder import build_symmetric_wisard
from wisardlib.utils import untie_by_first_class
from sklearn.metrics import accuracy_score, f1_score
from base_runner import (
    encoders,
    rams,
    datasets,
    discretize_labels,
    set_random_seeds,
    RAMConfig,
    load_dataset,
)

default_cfg = RAMConfig(name="DictRam", RAM_cls_name="dict")

configs = [
    # Count-Min-Sketch
    # RAMConfig(name="count-min-sketch", RAM_cls_name="count-min-sketch", RAM_cls_kwargs={"width": 100, "depth": 3}),
    # Heavy Hitters
    RAMConfig(
        name="heavy-hitters",
        RAM_cls_name="heavy-hitters",
        RAM_cls_kwargs={"num_hitters": 50, "width": 100, "depth": 3},
    ),
]


@jit(nopython=True, inline="always")
def int_to_binary_list(value: int, size: int):
    return [(value >> i) % 2 for i in range(size)]


def inspect_models(base_model, other_model, tuple_size: int, n_rams: int, y: int):
    n_rel, n_other_rel, n_diff = 0, 0, 0
    relevances, other_relevances = [], []

    for i in range(2**tuple_size):
        for j in range(n_rams):
            addr = np.array(int_to_binary_list(i, size=tuple_size), dtype=bool)
            relevance = base_model[y][j][addr]
            relevances.append(relevance)
            other_relevance = other_model[y][j][addr]
            other_relevances.append(other_relevance)
            diff = other_relevance - relevance
            if relevance > 0 or other_relevance > 0:
                print(
                    f"address: {addr*1}. diff: {diff}. relevance: {relevance}, other_relevance: {other_relevance}"
                )
            n_rel += relevance
            n_other_rel += other_relevance
            n_diff += abs(diff)

    print(f"relevance: {n_rel}, other_relevance: {n_other_rel}, max_diff: {n_diff}")
    print(f"relevances: {list(sorted([i for i in relevances if i > 0]))}")
    print(f"other relevances: {list(sorted([i for i in relevances if i > 0]))}")
    print(f"relevance factor: {len([i for i in relevances if i > 0])/2**tuple_size}")


def run_experiment(
    exp_name: str,
    output_path: Path,
    default_config=RAMConfig,
    ram_configs=List[RAMConfig],
    dataset_name: str = None,
    encoder_name: str = "thermometer",
    encoder_kwargs: dict = None,
    tuple_size=16,
    root_data_path: Path = ".",
    bleach: int = 1,
):
    exp_id = str(time.time())
    exp_dir = Path(output_path) / exp_name

    # Load dataset
    (x_train, y_train), (x_test, y_test) = load_dataset(
        dataset_name, Path(root_data_path)
    )

    y_train = discretize_labels(y_train)
    y_test = discretize_labels(y_test)

    # Encode dataset
    encoder_cls = encoders[encoder_name]
    encoder_kwargs = encoder_kwargs or dict()
    encoder = encoder_cls(**encoder_kwargs)
    logging.info("Encoding datasets...")
    encoder.fit(x_train)
    x_train = [x.ravel() for x in encoder.transform(x_train)]
    x_test = [x.ravel() for x in encoder.transform(x_test)]

    # Get params
    indices = list(range(len(x_train[0].ravel())))
    random.shuffle(indices)
    n_classes = max(y_train) + 1

    results = []
    times = []
    models = []
    n_rams = len(indices) // tuple_size

    for config_no, config in enumerate([default_config] + ram_configs):
        seed = set_random_seeds()
        logging.info(
            f"Running experiment (config={config_no+1}/{len(ram_configs)}) (run=1/1): {config}"
        )
        # logging.info("Creating model...")
        model = build_symmetric_wisard(
            RAM_cls=rams[config.RAM_cls_name],
            RAM_creation_kwargs=config.RAM_cls_kwargs,
            number_of_rams_per_discriminator=len(indices) // tuple_size,
            number_of_discriminators=n_classes,
            indices=indices,
            tuple_size=tuple_size,
            shuffle_indices=False,
        )

        model.fit(x_train, y_train)
        models.append(model)

    for i, model in enumerate(models):
        logging.info(f"Setting bleach to: {bleach}")
        models[0].bleach = bleach
        y_pred = models[0].predict(x_test)
        y_pred, ties = untie_by_first_class(y_pred)
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_micro = f1_score(y_test, y_pred, average="micro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")
        logging.info(
            f"[model {i}] Accuracy: {acc:.3f}, f1-score (weighted): {f1_weighted:.3f}, ties: {ties}"
        )

    for i, other_model in enumerate(models[1:]):
        print(f"----------Model {i+1} vs 0 -----------------")
        for y in range(n_classes):
            print(f"===== Class: {y}")
            inspect_models(
                models[0], other_model, tuple_size=tuple_size, n_rams=n_rams, y=y
            )
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="NormalExperimentRunner",
        description="Runs experiments in a dataset with a set of RAM configurations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-e",
        "--experiment-name",
        action="store",
        help="Name of the experiments",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-path",
        action="store",
        help="Output path to store results",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-d",
        "--dataset",
        action="store",
        choices=list(datasets.keys()),
        help="The dataset to use",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--encoder",
        action="store",
        choices=list(encoders.keys()),
        help="Encoder to use",
        default="thermometer",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--resolution",
        action="store",
        help="Encoder resolution",
        default=16,
        type=int,
        required=False,
    )
    parser.add_argument(
        "-t",
        "--tuple-size",
        action="store",
        help="Tuple size",
        default=16,
        type=int,
        required=False,
    )
    parser.add_argument(
        "-l",
        "--level",
        action="store",
        help="Logging level",
        default=20,
        type=int,
        required=False,
    )
    parser.add_argument(
        "--root-data-dir",
        action="store",
        help="Root data directory",
        default=".",
        type=str,
        required=False,
    )
    parser.add_argument(
        "-b",
        "--bleach",
        action="store",
        help="Bleach values",
        default=1,
        type=int,
        required=False,
    )

    args = parser.parse_args()
    print(args)

    formatter = "[%(asctime)s] [%(levelname)s]: %(message)s"
    logging.basicConfig(format=formatter, level=args.level)

    run_experiment(
        exp_name=args.experiment_name,
        output_path=args.output_path,
        dataset_name=args.dataset,
        default_config=default_cfg,
        ram_configs=configs,
        encoder_name=args.encoder,
        encoder_kwargs=dict(resolution=args.resolution),
        tuple_size=args.tuple_size,
        bleach=args.bleach,
        root_data_path=args.root_data_dir,
    )

    print("Done")
