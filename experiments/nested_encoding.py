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

from sklearn.model_selection import GroupKFold

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

from wisardlib.encoders.thermometer import NestedThermometerEncoder, ThermometerEncoder

def run_experiment(
    exp_name: str,
    output_path: Path,
    dataset_name: str,
    resolution_1: int,
    resolution_2: int,
    shuffle_first: bool = True,
    tuple_size=16,
    root_data_path: Path = ".",
    bleach: int = 1,
    n_models: int = 2
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
    encoder = NestedThermometerEncoder(resolution=resolution_1, resolution_2=resolution_2, shuffle=shuffle_first)
    # encoder = ThermometerEncoder(resolution=resolution_1)
    logging.info("Encoding datasets...")
    encoder.fit(x_train)
    x_train = np.asarray([x.ravel() for x in encoder.transform(x_train)])
    x_test = np.asarray([x.ravel() for x in encoder.transform(x_test)])

    # Get params
    samples_indices = list(range(len(x_train)))
    indices = list(range(len(x_train[0].ravel())))
    n_classes = max(y_train) + 1

    results = []
    times = []
    models = []
    n_rams = len(indices) // tuple_size

    seed = set_random_seeds()

    # logging.info(
    #     f"Running experiment (config={config_no+1}/{len(ram_configs)}) (model={model_no}/{n_models}): {config}"
    # )
    # logging.info("Creating model...")
    model = build_symmetric_wisard(
        RAM_cls=rams["dict"],
        RAM_creation_kwargs=dict(),
        number_of_rams_per_discriminator=len(indices) // tuple_size,
        number_of_discriminators=n_classes,
        indices=indices,
        tuple_size=tuple_size,
        shuffle_indices=False
    )

    train_start = time.time()
    model.fit(x_train, y_train)
    train_end = time.time()

    # print(f"Setting bleach to: {bleach}")
    model.bleach = bleach
    predict_start = time.time()
    y_pred = model.predict(x_test)
    predict_end = time.time()
    y_pred, ties = untie_by_first_class(y_pred, use_tqdm=True)

    model_size = model.size()
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_micro = f1_score(y_test, y_pred, average="micro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    print(
        f" --- Accuracy: {acc:.3f}, f1-score (weighted): {f1_weighted:.3f}, ties: {ties}, size: {model_size}"
    )

    # results.append(
    #     {
    #         "bleach": bleach,
    #         "accuracy": acc,
    #         "f1 weighted": f1_weighted,
    #         "f1 macro": f1_macro,
    #         "f1 micro": f1_micro,
    #         "ties": ties,
    #         "model no": model_no,
    #         "train time": train_end - train_start,
    #         "predict time": predict_end - predict_start,
    #         "ram name": config.name,
    #         "ram kwargs": json.dumps(config.RAM_cls_kwargs),
    #         "tuple size": tuple_size,
    #         "dataset name": dataset_name,
    #         "encoder": encoder_name,
    #         "encoder kwargs": json.dumps(encoder_kwargs),
    #         "experiment name": exp_name,
    #         "model size": model.size(),
    #         "train samples": len(_y_train),
    #         "test samples": len(y_test),
    #         "classes": n_classes,
    #         "rams per discriminator": len(indices) // tuple_size,
    #         "discriminators": n_classes,
    #         "seed": seed,
    #         "indices": len(indices),
    #     }
    # )




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
        "--resolution",
        action="store",
        help="Encoder resolution",
        default=16,
        type=int,
        required=False,
    )
    parser.add_argument(
        "--resolution-2",
        action="store",
        help="Encoder2 resolution",
        default=8,
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
        resolution_1=args.resolution,
        resolution_2=args.resolution_2,
        shuffle_first=True,
        tuple_size=args.tuple_size,
        bleach=args.bleach,
        root_data_path=args.root_data_dir
    )

    print("Done")
