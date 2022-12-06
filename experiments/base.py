from pathlib import Path
import numpy as np
import pandas as pd
import random
import time
import logging
from dataclasses import dataclass, field
from typing import List

from wisardlib.encoders.thermometer import ThermometerEncoder, DistributiveThermometerEncoder
from wisardlib.rams.dict_ram import DictRAM
from wisardlib.rams.bloom_filter_ram import CountingBloomFilterRAM, CountMinSketchRAM, CountingCuckooRAM, HeavyHittersRAM, StreamThresholdRAM
from wisardlib.builder import build_symmetric_wisard
from wisardlib.utils import untie_by_first_class
from sklearn.metrics import accuracy_score, f1_score

encoders = {
    "thermometer": ThermometerEncoder,
    "distributive-thermometer": DistributiveThermometerEncoder
}

rams = {
    "dict": DictRAM,
    "count-bloom": CountingBloomFilterRAM,
    "count-min-sketch": CountMinSketchRAM,
    "count-cuckoo": CountingCuckooRAM,
    "heavy-hitters": HeavyHittersRAM,
    "stream-threshold": StreamThresholdRAM
}

datasets = {
    "iris": {
        "path": Path("data/iris/data.pkl")
    },

    "wine": { #Z
        "path": Path("data/wine/data.pkl")
    },

    "breast_cancer": { # Z
        "path": Path("data/breast_cancer/data.pkl")
    },

    "ecoli": { # NF
        "path": Path("data/ecoli/data.pkl")
    },

    "letter": {
        "path": Path("data/letter/data.pkl")
    },

    "satimage": {
        "path": Path("data/satimage/data.pkl")
    },

    "segment": {
        "path": Path("data/segment/data.pkl")
    },

    "glass": {
        "path": Path("data/glass/data.pkl")
    },

    "mnist": {
        "path": Path("data/mnist/data.pkl")
    }
}


@dataclass
class RAMConfig:
    name: str
    RAM_cls_name: str
    RAM_cls_kwargs: dict = field(default_factory=dict)

def load_dataset(name: str) -> tuple:
    data_path = datasets[name]["path"]
    (x_train, y_train), (x_test, y_test) = np.load(data_path, allow_pickle=True)
    return (x_train, y_train), (x_test, y_test)

def run_experiment(
    exp_name: str,
    output_path: Path,
    ram_configs = List[RAMConfig],
    dataset_name: str = None,
    encoder_name: str = "thermometer",
    encoder_kwargs: dict = None,
    tuple_size = 16,
    number_runs = 3,
    bleach: int | List[int] = 1
):
    exp_id = str(time.time())
    exp_dir = Path(output_path) / exp_name

    if isinstance(bleach, int):
        bleach = [bleach]

    # Load dataset
    (x_train, y_train), (x_test, y_test) = load_dataset(dataset_name)

    # Encode dataset
    encoder_cls = encoders[encoder_name]
    encoder_kwargs = encoder_kwargs or dict()
    encoder = encoder_cls(**encoder_kwargs)
    logging.info("Encoding datasets...")
    encoder.fit(x_train)
    x_train = [x.ravel() for x in encoder.transform(x_train)]
    x_test  = [x.ravel() for x in encoder.transform(x_test)]

    # Get params
    indices = list(range(len(x_train[0].ravel())))
    n_classes = len(np.unique(y_train))+2

    results = []

    for config in ram_configs:
        for run_no in range(number_runs):
            logging.info("Creating model...")
            model = build_symmetric_wisard(
                RAM_cls=rams[config.RAM_cls_name],
                RAM_creation_kwargs=config.RAM_cls_kwargs,
                number_of_rams_per_discriminator=len(indices)//tuple_size,
                number_of_discriminators=n_classes,
                indices=indices,
                tuple_size=tuple_size,
                shuffle_indices=True
            )

            logging.info("Training model...")
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

                acc = accuracy_score(y_test, y_pred)
                f1_macro = f1_score(y_test, y_pred, average="macro")
                f1_micro = f1_score(y_test, y_pred, average="micro")
                f1_weighted = f1_score(y_test, y_pred, average="weighted")

                results.append({
                    "bleach": bleach_val,
                    "accuracy": acc,
                    "f1 weighted": f1_weighted,
                    "f1 macro": f1_macro,
                    "f1 micro": f1_micro,
                    "ties": ties,
                    "run": run_no+1,
                    "train time": train_end - train_start,
                    "predict time": predict_end - predict_start,
                    "ram name": config.name,
                    "ram kwargs": str(config.RAM_cls_kwargs),
                    "tuple size": tuple_size,
                    "dataset name": dataset_name,
                    "encoder": encoder_name,
                    "encoder kwargs": str(encoder_kwargs),
                    "experiment name": exp_name,
                    "size": model.size()
                })

    df = pd.DataFrame(results)
    exp_dir.mkdir(exist_ok=True, parents=True)
    exp_path = exp_dir / f"{exp_id}.csv"
    df.to_csv(exp_path, index=False)

    return df
