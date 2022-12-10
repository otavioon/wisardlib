from pathlib import Path
import numpy as np
import pandas as pd
import random
import time
import logging
import json
from dataclasses import dataclass, field
from typing import List

from wisardlib.encoders.thermometer import (
    ThermometerEncoder,
    DistributiveThermometerEncoder,
)
from wisardlib.rams.dict_ram import DictRAM
from wisardlib.rams.bloom_filter_ram import (
    CountingBloomFilterRAM,
    CountMinSketchRAM,
    CountingCuckooRAM,
    HeavyHittersRAM,
    StreamThresholdRAM,
)
from wisardlib.builder import build_symmetric_wisard
from wisardlib.utils import untie_by_first_class
from sklearn.metrics import accuracy_score, f1_score

encoders = {
    "thermometer": ThermometerEncoder,
    "distributive-thermometer": DistributiveThermometerEncoder,
}

rams = {
    "dict": DictRAM,
    "count-bloom": CountingBloomFilterRAM,
    "count-min-sketch": CountMinSketchRAM,
    "count-cuckoo": CountingCuckooRAM,
    "heavy-hitters": HeavyHittersRAM,
    "stream-threshold": StreamThresholdRAM,
}

datasets = {
    "iris": {"path": Path("experiments/data/iris/data.pkl")}, # OK
    "wine": {"path": Path("experiments/data/wine/data.pkl")},  # OK
    "breast_cancer": {"path": Path("experiments/data/breast_cancer/data.pkl")},  # OK
    "ecoli": {"path": Path("experiments/data/ecoli/data.pkl")},  #
    "letter": {"path": Path("experiments/data/letter/data.pkl")}, # OK
    "satimage": {"path": Path("experiments/data/satimage/data.pkl")}, # OK
    "segment": {"path": Path("experiments/data/segment/data.pkl")}, #~OK
    "glass": {"path": Path("experiments/data/glass/data.pkl")}, # OK
    "mnist": {"path": Path("experiments/data/mnist/data.pkl")}, # OK
    "vehicle": {"path": Path("experiments/data/vehicle/data.pkl")}, # OK
    "motion_sense": {"path": Path("experiments/data/motion_sense/data.pkl")}, # OK
    "sensorless_drive": {"path": Path("experiments/data/sensorless_drive/data.pkl")}, # OK
}


@dataclass
class RAMConfig:
    name: str
    RAM_cls_name: str
    RAM_cls_kwargs: dict = field(default_factory=dict)


def load_dataset(name: str, root_data_path: Path) -> tuple:
    data_path = datasets[name]["path"]
    data_path = root_data_path / data_path
    (x_train, y_train), (x_test, y_test) = np.load(data_path, allow_pickle=True)
    return (x_train, y_train), (x_test, y_test)


def discretize_labels(labels: np.ndarray):
    single_labels = list(set(labels))
    return np.array([single_labels.index(l) for l in labels])


def set_random_seeds(seed: int = None):
    if not seed:
        seed = int(time.time())
    random.seed(seed)
    np.random.seed(seed)
    return seed


def run_experiment(
    exp_name: str,
    output_path: Path,
    ram_configs=List[RAMConfig],
    dataset_name: str = None,
    encoder_name: str = "thermometer",
    encoder_kwargs: dict = None,
    tuple_size=16,
    number_runs=3,
    bleach: int | List[int] = 1,
    root_data_path: Path = ".",
):
    exp_id = str(time.time())
    exp_dir = Path(output_path) / exp_name

    if isinstance(bleach, int):
        bleach = [bleach]

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
    indices = len(x_train[0].ravel())
    n_classes = max(y_train) + 1

    results = []
    times = []

    for config_no, config in enumerate(ram_configs):
        for run_no in range(number_runs):
            seed = set_random_seeds()
            exp_start_time = time.time()
            logging.info(
                f"Running experiment (config={config_no+1}/{len(ram_configs)}) (run={run_no+1}/{number_runs}): {config}"
            )
            # logging.info("Creating model...")
            model = build_symmetric_wisard(
                RAM_cls=rams[config.RAM_cls_name],
                RAM_creation_kwargs=config.RAM_cls_kwargs,
                number_of_rams_per_discriminator=indices // tuple_size,
                number_of_discriminators=n_classes,
                indices=indices,
                tuple_size=tuple_size,
                shuffle_indices=True,
            )

            # logging.info("Training model...")
            train_start = time.time()
            model.fit(x_train, y_train)
            train_end = time.time()

            for bleach_val in bleach:
                logging.info(f"Setting bleach to: {bleach_val}")
                model.bleach = bleach_val
                predict_start = time.time()
                y_pred = model.predict(x_test)
                predict_end = time.time()
                y_pred, ties = untie_by_first_class(y_pred)

                model_size = model.size()
                acc = accuracy_score(y_test, y_pred)
                f1_macro = f1_score(y_test, y_pred, average="macro")
                f1_micro = f1_score(y_test, y_pred, average="micro")
                f1_weighted = f1_score(y_test, y_pred, average="weighted")
                logging.info(
                    f"Accuracy: {acc:.3f}, f1-score (weighted): {f1_weighted:.3f}, ties: {ties}, size: {model_size}"
                )

                results.append(
                    {
                        "bleach": bleach_val,
                        "accuracy": acc,
                        "f1 weighted": f1_weighted,
                        "f1 macro": f1_macro,
                        "f1 micro": f1_micro,
                        "ties": ties,
                        "run": run_no + 1,
                        "train time": train_end - train_start,
                        "predict time": predict_end - predict_start,
                        "ram name": config.name,
                        "ram kwargs": json.dumps(config.RAM_cls_kwargs),
                        "tuple size": tuple_size,
                        "dataset name": dataset_name,
                        "encoder": encoder_name,
                        "encoder kwargs": json.dumps(encoder_kwargs),
                        "experiment name": exp_name,
                        "model size": model.size(),
                        "train samples": len(y_train),
                        "test samples": len(y_test),
                        "classes": n_classes,
                        "rams per discriminator": indices // tuple_size,
                        "discriminators": n_classes,
                        "seed": seed,
                        "indices": indices,
                    }
                )
            exp_end_time = time.time()
            times.append(exp_end_time - exp_start_time)
            logging.info(
                f"The run took {exp_end_time-exp_start_time:.3f} seconds (estimated remaining time: {len(ram_configs)*number_runs*np.mean(times)-config_no*number_runs*np.mean(times)-(run_no+1)*np.mean(times):.2f} seconds...)\n"
            )

    df = pd.DataFrame(results)
    exp_dir.mkdir(exist_ok=True, parents=True)
    exp_path = exp_dir / f"{exp_id}.csv"
    df.to_csv(exp_path, index=False)
    print(f"Results saved to: {exp_path}")

    return df
