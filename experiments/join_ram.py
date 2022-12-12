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

configs = [
    #Dict RAM
    RAMConfig(name="DictRam", RAM_cls_name="count-bloom"),

    RAMConfig(
        name="count-bloom",
        RAM_cls_name="count-bloom",
        RAM_cls_kwargs={"est_elements": 1000, "false_positive_rate": 0.05},
    ),

    RAMConfig(
        name="count-min-sketch",
        RAM_cls_name="count-min-sketch",
        RAM_cls_kwargs={"width": 1000, "depth": 5},
    ),

    # RAMConfig(
    #     name="count-cuckoo",
    #     RAM_cls_name="count-cuckoo",
    #     RAM_cls_kwargs={"capacity": 1000, "bucket_size": 4},
    # ),

    # RAMConfig(
    #     name="heavy-hitters",
    #     RAM_cls_name="heavy-hitters",
    #     RAM_cls_kwargs={"num_hitters": 100, "width": 1000, "depth": 5},
    # ),

    # RAMConfig(
    #     name="stream-threshold",
    #     RAM_cls_name="stream-threshold",
    #     RAM_cls_kwargs={"threshold": 100, "width": 1000, "depth": 5},
    # ),
]


def run_experiment(
    exp_name: str,
    output_path: Path,
    ram_configs=List[RAMConfig],
    dataset_name: str = None,
    encoder_name: str = "thermometer",
    encoder_kwargs: dict = None,
    tuple_size=16,
    root_data_path: Path = ".",
    bleach: int = 1,
    n_models: int = 2
):
    assert n_models > 1, "must be at least 2 models to join"

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
    x_train = np.asarray([x.ravel() for x in encoder.transform(x_train)])
    x_test = np.asarray([x.ravel() for x in encoder.transform(x_test)])

    # Get params
    samples_indices = list(range(len(x_train)))
    indices = list(range(len(x_train[0].ravel())))
    random.shuffle(samples_indices)
    random.shuffle(indices)
    n_classes = max(y_train) + 1

    results = []
    times = []
    models = []
    n_rams = len(indices) // tuple_size

    for config_no, config in enumerate(ram_configs):
        original_model = None
        partial_models = []

        for model_no in range(n_models+1):
            seed = set_random_seeds()

            if model_no != n_models:
                split_size = len(x_train)//n_models
                _x_train = x_train[samples_indices[model_no*split_size:min((model_no+1)*split_size, len(x_train))]]
                _y_train = y_train[samples_indices[model_no*split_size:min((model_no+1)*split_size, len(y_train))]]
            else:
                split_size = len(x_train)
                _x_train = x_train
                _y_train = y_train

            # logging.info(
            #     f"Running experiment (config={config_no+1}/{len(ram_configs)}) (model={model_no}/{n_models}): {config}"
            # )
            # logging.info("Creating model...")
            model = build_symmetric_wisard(
                RAM_cls=rams[config.RAM_cls_name],
                RAM_creation_kwargs=config.RAM_cls_kwargs,
                number_of_rams_per_discriminator=len(indices) // tuple_size,
                number_of_discriminators=n_classes,
                indices=indices,
                tuple_size=tuple_size,
                shuffle_indices=False
            )

            model.use_tqdm = False

            train_start = time.time()
            model.fit(_x_train, _y_train)
            train_end = time.time()

            # print(f"Setting bleach to: {bleach}")
            model.bleach = bleach
            predict_start = time.time()
            y_pred = model.predict(x_test)
            predict_end = time.time()
            y_pred, ties = untie_by_first_class(y_pred, use_tqdm=False)

            model_size = model.size()
            acc = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average="macro")
            f1_micro = f1_score(y_test, y_pred, average="micro")
            f1_weighted = f1_score(y_test, y_pred, average="weighted")
            print(
                f" --- [{model_no}/{n_models}] Accuracy: {acc:.3f}, f1-score (weighted): {f1_weighted:.3f}, ties: {ties}, size: {model_size}"
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

            if model_no != n_models:
                partial_models.append(model)
            else:
                original_model = model

        for model in partial_models[1:]:
            partial_models[0].join(model)

        print(f"------------ Summary [{config.name}] ------------")

        print(f"Using bleach value of: {bleach}")

        # --------- Joinned model -----------
        model = partial_models[0]
        model.use_tqdm = False
        model.bleach = bleach
        predict_start = time.time()
        y_pred = model.predict(x_test)
        predict_end = time.time()
        y_pred, ties = untie_by_first_class(y_pred, use_tqdm=False)

        model_size = model.size()
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_micro = f1_score(y_test, y_pred, average="micro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")
        print(
            f"[JOIN Model] Accuracy: {acc:.3f}, f1-score (weighted): {f1_weighted:.3f}, ties: {ties}, size: {model_size}"
        )

        # --------- Original model ---------
        model = original_model
        model.use_tqdm = False
        model.bleach = bleach
        predict_start = time.time()
        y_pred = model.predict(x_test)
        predict_end = time.time()
        y_pred, ties_original = untie_by_first_class(y_pred, use_tqdm=False)

        model_size_original = model.size()
        acc_original = accuracy_score(y_test, y_pred)
        f1_macro_original = f1_score(y_test, y_pred, average="macro")
        f1_micro_original = f1_score(y_test, y_pred, average="micro")
        f1_weighted_original = f1_score(y_test, y_pred, average="weighted")
        print(
            f"[Original Model] Accuracy: {acc_original:.3f}, f1-score (weighted): {f1_weighted_original:.3f}, ties: {ties_original}, size: {model_size_original}"
        )

        print(f"Acc loss: {acc_original-acc:.3f}, f1-score (weighted) loss: {f1_weighted_original-f1_weighted:.3f}, ties loss: {ties_original-ties}, size loss: {model_size_original-model_size}")

        print("-------------------------------")
        # results.append(
        #     {
        #         "bleach": bleach,
        #         "accuracy": acc,
        #         "f1 weighted": f1_weighted,
        #         "f1 macro": f1_macro,
        #         "f1 micro": f1_micro,
        #         "ties": ties,
        #         "model no": n_models,
        #         "train time": 0,
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
    # for models in models:



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
    parser.add_argument(
        "-m",
        "--models",
        action="store",
        help="Number of models to join",
        default=2,
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
        ram_configs=configs,
        encoder_name=args.encoder,
        encoder_kwargs=dict(resolution=args.resolution),
        tuple_size=args.tuple_size,
        bleach=args.bleach,
        root_data_path=args.root_data_dir,
        n_models=args.models
    )

    print("Done")
