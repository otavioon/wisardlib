import argparse
from ast import Dict
from collections import defaultdict
from pathlib import Path
import random
import time
from typing import Optional

import numpy as np
from ray import tune
import optuna
from ray.tune.search.optuna import OptunaSearch
from ray.train import RunConfig
from ray.tune import TuneConfig
from datetime import datetime

from sklearn.model_selection import train_test_split

from wisardlib.encoders.thermometer import (
    ThermometerEncoder,
    DistributiveThermometerEncoder,
)
from wisardlib.rams.dict_ram import DictRAM
from wisardlib.rams.bloom_filter_ram import (
    CountingBloomFilterRAM,
    CountMinSketchRAM,
    CountMeanSketchRAM,
    CountMeanMinSketchRAM,
    CountingCuckooRAM,
    HeavyHittersRAM,
    StreamThresholdRAM,
)
from ray.tune.search import ConcurrencyLimiter

from wisardlib.builder import build_symmetric_wisard
from wisardlib.utils import untie_by_first_class
from sklearn.metrics import accuracy_score, f1_score
from time import perf_counter

from ray import tune
from ray.train import FailureConfig


encoders_cls = {
    "thermometer": ThermometerEncoder,
    "distributive-thermometer": DistributiveThermometerEncoder,
}

rams_cls = {
    "dict": DictRAM,
    "count-bloom": CountingBloomFilterRAM,
    "count-min-sketch": CountMinSketchRAM,
    "count-mean-sketch": CountMeanSketchRAM,
    "count-mean-min-sketch": CountMeanMinSketchRAM,
    "heavy-hitters": HeavyHittersRAM,
    "stream-threshold": StreamThresholdRAM,
    "count-cuckoo": CountingCuckooRAM,
}

datasets = {
    "iris": {"path": Path("iris/data.pkl")},  # OK
    "wine": {"path": Path("wine/data.pkl")},  # OK
    "breast_cancer": {"path": Path("breast_cancer/data.pkl")},  # OK
    "ecoli": {"path": Path("ecoli/data.pkl")},  #
    "letter": {"path": Path("letter/data.pkl")},  # OK
    "satimage": {"path": Path("satimage/data.pkl")},  # OK
    "segment": {"path": Path("segment/data.pkl")},  # ~OK
    "glass": {"path": Path("glass/data.pkl")},  # OK
    "mnist": {"path": Path("mnist/data.pkl")},  # OK
    "vehicle": {"path": Path("vehicle/data.pkl")},  # OK
    "motion_sense": {"path": Path("motion_sense/data.pkl")},  # OK
    "sensorless_drive": {"path": Path("sensorless_drive/data.pkl")},  # OK
    "olivetti": {"path": Path("olivetti/data.pkl")},  # OK
    "optical_handwritten": {"path": Path("optical_handwritten/data.pkl")},  # OK
    "image_segmentation": {"path": Path("image_segmentation/data.pkl")},  # OK
    "sepsis": {"path": Path("sepsis/data.pkl")},  # OK
    "rice": {"path": Path("rice/data.pkl")},  # OK
    "yeast": {"path": Path("yeast/data.pkl")},  # OK
    "dry_bean": {"path": Path("dry_bean/data.pkl")},  # OK
}


def load_dataset(name: str, root_data_path: Path) -> tuple:
    data_path = datasets[name]["path"]
    data_path = root_data_path / data_path
    (x_train, y_train), (x_test, y_test) = np.load(data_path, allow_pickle=True)
    return (x_train, y_train), (x_test, y_test)


def discretize_labels(labels: np.ndarray):
    single_labels = list(set(labels))
    return np.array([single_labels.index(l) for l in labels])


class Timer:
    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.start



class BaseEperiment(tune.Trainable):
    RAM_name = ""

    @staticmethod
    def get_search_space():
        raise NotImplementedError

    def _get_ram_config(self, config: Dict) -> Dict:
        raise NotImplementedError

    def _get_metrics(self, metrics, y_pred, y_true, ties, stage: str):
        metrics["ties"].append(ties)
        metrics["accuracy"].append(
            accuracy_score(y_true, y_pred),
        )
        metrics["f1 weighted"].append(
            f1_score(y_true, y_pred, average="weighted")
        )     
        metrics["size"].append(self.model.size())

        result = {
            # Ties
            f"{stage}_ties": ties,
            f"{stage}_ties_mean": np.mean(metrics["ties"]),
            f"{stage}_ties_std": np.std(metrics["ties"]),
            
            # Accuracy
            f"{stage}_accuracy": metrics["accuracy"][-1],
            f"{stage}_accuracy_mean": np.mean(np.array(metrics["accuracy"])),
            f"{stage}_accuracy_std": np.std(np.array(metrics["accuracy"])),
            
            # f1 weighted
            f"{stage}_f1 weighted": metrics["f1 weighted"][-1],
            f"{stage}_f1 weighted_mean": np.mean(np.array(metrics["f1 weighted"])),
            f"{stage}_f1 weighted_std": np.std(
                np.array(metrics["f1 weighted"])
            ),
            
            # Size
            f"{stage}_model size": metrics["size"][-1],
            f"{stage}_model size_mean": np.mean(np.array(metrics["size"])),
            f"{stage}_model size_std": np.std(np.array(metrics["size"])),
        }
        
        return result

    def setup(self, config):
        self.val_metrics = defaultdict(list)
        self.test_metrics = defaultdict(list)

        self.root_data_dir = Path(config["root_data_dir"])
        self.dataset_name = config["dataset_name"]
        self.tuple_size = (
            config["resolution"] // config["tuple_resolution_factor"]
        )
        self.encoder_name = config["encoder"]
        self.encoder_kwargs = {"resolution": config["resolution"]}
        self.encoder = encoders_cls[self.encoder_name](**self.encoder_kwargs)

        (self.x_train, self.y_train), (self.x_test, self.y_test) = load_dataset(
            self.dataset_name, self.root_data_dir
        )
        # Partition in x_train, x_val, y_train, y_val
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            self.x_train,
            self.y_train,
            test_size=0.2,
            random_state=42,
            stratify=self.y_train,
        )

        # Fit encoder on train data
        self.encoder.fit(self.x_train)
        # Transform data
        self.x_train = [x.ravel() for x in self.encoder.transform(self.x_train)]
        self.x_val = [x.ravel() for x in self.encoder.transform(self.x_val)]
        self.x_test = [x.ravel() for x in self.encoder.transform(self.x_test)]
        self.y_train = discretize_labels(self.y_train)
        self.y_val = discretize_labels(self.y_val)
        self.y_test = discretize_labels(self.y_test)
        # Indices and classes
        self.indices = len(self.x_train[0].ravel())
        self.n_classes = max(self.y_train) + 1
        # RAM
        self.ram_cls = rams_cls[self.RAM_name]
        self.ram_kwargs = self._get_ram_config(config)
        # Other parameters
        self.bleach = config["bleach"]
        self.number_of_rams_per_discriminator = self.indices // self.tuple_size

    def step(self):
        self.model = build_symmetric_wisard(
            RAM_cls=self.ram_cls,
            RAM_creation_kwargs=self.ram_kwargs,
            number_of_rams_per_discriminator=self.number_of_rams_per_discriminator,
            number_of_discriminators=self.n_classes,
            indices=self.indices,
            tuple_size=self.tuple_size,
            shuffle_indices=True,
            use_tqdm=False,
        )

        # Fit model on train data
        with Timer() as t_train_time:
            self.model.fit(self.x_train, self.y_train)

        # Predict with validation data
        self.model.bleach = self.bleach
        
        with Timer() as t_val_predict_time:
            y_pred = self.model.predict(self.x_val)
            y_pred, ties = untie_by_first_class(y_pred, use_tqdm=False)
            
        metrics = self._get_metrics(
            metrics=self.val_metrics,
            y_pred=y_pred,
            y_true=self.y_val,
            ties=ties,
            stage="val",
        )
        metrics["ram"] = self.RAM_name
        metrics["train time"] = t_train_time.time
        metrics["val_predict time"] = t_val_predict_time.time
        metrics["train_samples"] = len(self.x_train)
        metrics["val_samples"] = len(self.x_val)
        metrics["test_samples"] = len(self.x_test)
        metrics["classes"] = self.n_classes
        metrics["rams per discriminator"] = self.number_of_rams_per_discriminator
        metrics["discriminators"] = self.n_classes
        metrics["indices"] = self.indices
        
        # Add validation samples
        self.model.fit(self.x_val, self.y_val)
        
        # Predict with test data
        self.model.bleach = self.bleach
        with Timer() as t_test_predict_time:
            y_pred = self.model.predict(self.x_test)
            y_pred, ties = untie_by_first_class(y_pred, use_tqdm=False)
            
        test_metrics = self._get_metrics(
            metrics=self.test_metrics,
            y_pred=y_pred,
            y_true=self.y_test,
            ties=ties,
            stage="test",
        )
        
        metrics.update(test_metrics)
        metrics["test_predict time"] = t_test_predict_time.time
        return metrics

    def cleanup(self):
        pass

    def save_checkpoint(self, checkpoint_dir: str) -> Optional[Dict]:
        pass

    def load_checkpoint(self, checkpoint: Optional[Dict]):
        pass


class DictExperiment(BaseEperiment):
    RAM_name = "dict"

    @staticmethod
    def get_search_space():
        return {}

    def _get_ram_config(self, config: Dict) -> Dict:
        return {}


class CountingBloomFilterExperiment(BaseEperiment):
    RAM_name = "count-bloom"

    def _get_ram_config(self, config: Dict) -> Dict:
        return {
            "est_elements": config["est_elements"],
            "false_positive_rate": config["false_positive_rate"],
        }

    @staticmethod
    def get_search_space():
        return {
            "est_elements": optuna.distributions.IntDistribution(
                10, 1000, log=False, step=10
            ),
            "false_positive_rate": optuna.distributions.FloatDistribution(
                0.01, 0.9, log=True
            ),
        }


class CountMinSketchExperiment(BaseEperiment):
    RAM_name = "count-min-sketch"

    def _get_ram_config(self, config: Dict) -> Dict:
        return {"width": config["width"], "depth": config["depth"]}

    @staticmethod
    def get_search_space():
        return {
            "width": optuna.distributions.IntDistribution(
                1, 500, log=False, step=3
            ),
            "depth": optuna.distributions.IntDistribution(1, 5),
        }


class CountMeanSketchExperiment(CountMinSketchExperiment):
    RAM_name = "count-mean-sketch"


class CountMeanMinSketchExperiment(CountMinSketchExperiment):
    RAM_name = "count-mean-min-sketch"


class CountingCuckooExperiment(BaseEperiment):
    RAM_name = "count-cuckoo"

    def _get_ram_config(self, config: Dict) -> Dict:
        return {
            "capacity": config["capacity"],
            "bucket_size": config["bucket_size"],
        }

    @staticmethod
    def get_search_space():
        return {
            "capacity": optuna.distributions.IntDistribution(
                10, 1000, log=False, step=10
            ),
            "bucket_size": optuna.distributions.IntDistribution(1, 10),
        }


class HeavyHittersExperiment(BaseEperiment):
    RAM_name = "heavy-hitters"

    def _get_ram_config(self, config: Dict) -> Dict:
        return {
            "num_hitters": config["num_hitters"],
            "width": config["width"],
            "depth": config["depth"],
        }

    @staticmethod
    def get_search_space():
        return {
            "num_hitters": optuna.distributions.IntDistribution(
                5, 1000, log=False, step=5
            ),
            "width": optuna.distributions.IntDistribution(
                1, 500, log=False, step=3
            ),
            "depth": optuna.distributions.IntDistribution(1, 5),
        }


class StreamThresholdExperiment(BaseEperiment):
    RAM_name = "stream-threshold"

    def _get_ram_config(self, config: Dict) -> Dict:
        return {
            "threshold": config["threshold"],
            "width": config["width"],
            "depth": config["depth"],
        }

    @staticmethod
    def get_search_space():
        return {
            "threshold": optuna.distributions.IntDistribution(
                5, 1000, log=False, step=5
            ),
            "width": optuna.distributions.IntDistribution(
                1, 500, log=False, step=3
            ),
            "depth": optuna.distributions.IntDistribution(1, 5),
        }


def main(
    dataset: str,
    ram: str,
    root_data_dir: Path,
    metric: str,
    output_dir: Path,
    experiment_name: str,
    resolution_min: int,
    resolution_max: int,
    bleach_min: int,
    bleach_max: int,
    cpus: int,
    budget: int,
    num_samples: int = 10000,
    concurrent_trials: int = None,
):
    experiment_map = {
        "dict": DictExperiment,
        "count-bloom": CountingBloomFilterExperiment,
        "count-min-sketch": CountMinSketchExperiment,
        "count-mean-sketch": CountMeanSketchExperiment,
        "count-mean-min-sketch": CountMeanMinSketchExperiment,
        "heavy-hitters": HeavyHittersExperiment,
        "stream-threshold": StreamThresholdExperiment,
        "count-cuckoo": CountingCuckooExperiment,
    }

    experiment_cls = experiment_map[ram]

    # ------  Search space -------
    space = {
        "root_data_dir": optuna.distributions.CategoricalDistribution(
            [str(root_data_dir)]
        ),
        "dataset_name": optuna.distributions.CategoricalDistribution([dataset]),
        "resolution": optuna.distributions.IntDistribution(
            resolution_min, resolution_max, step=1
        ),
        "bleach": optuna.distributions.IntDistribution(
            bleach_min, bleach_max, step=1, log=True
        ),
        "tuple_resolution_factor": optuna.distributions.CategoricalDistribution(
            [1, 2]
        ),
        "encoder": optuna.distributions.CategoricalDistribution(
            ["thermometer", "distributive-thermometer"]
        ),
    }

    space.update(experiment_cls.get_search_space())

    # ------  Searcher -------
    searcher = OptunaSearch(
        space,
        metric=[metric, "val_model size_mean"],
        mode=["max", "min"],
    )

    # ------  Tune config -------

    tune_config = TuneConfig(
        search_alg=searcher,
        time_budget_s=budget,
        num_samples=num_samples,
        max_concurrent_trials=concurrent_trials,
    )

    # ------  Run config -------

    run_config = RunConfig(
        name=experiment_name,
        storage_path=output_dir,
        stop={"training_iteration": 3},
        failure_config=FailureConfig(max_failures=0),
    )

    # ------  Tuner -------

    if cpus is not None:
        print(f"*** Tuning with {cpus} cpus")
        experiment_cls = tune.with_resources(
            experiment_cls, resources={"cpu": cpus}
        )

    print("*** Tuning...")
    tuner = tune.Tuner(
        experiment_cls,
        tune_config=tune_config,
        run_config=run_config,
    )

    results_grid = tuner.fit()

    results = results_grid.get_dataframe()
    results.to_csv(output_dir / f"{experiment_name}.csv", index=False)
    print(f"Results saved at {output_dir / f'{experiment_name}.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run experiment with ray tune",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        choices=list(datasets.keys()),
        type=str,
        required=True,
        help="Name of the dataset to use",
    )
    parser.add_argument(
        "--ram",
        choices=list(rams_cls.keys()),
        type=str,
        required=True,
        help="RAM to use",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="accuracy",
        required=False,
        help="Performance metric to optimize",
    )
    parser.add_argument(
        "--root-data-dir",
        type=str,
        required=True,
        help="Root directory of the data. Inside this directory there should be folders with the name of the datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        default="ray_results",
        help="Output directory to store results. It will be at <output-dir>/<dataset_name>/<experiment-name>",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        required=False,
        default=None,
        help="Name of the experiment (output csv file). If None, time.time will be used",
    )
    parser.add_argument(
        "--resolution-min",
        type=int,
        default=4,
        required=False,
        help="Minimum resolution of the encoder",
    )
    parser.add_argument(
        "--resolution-max",
        type=int,
        default=64,
        required=False,
        help="Maximum resolution of the encoder",
    )
    parser.add_argument(
        "--bleach-min",
        type=int,
        default=1,
        required=False,
        help="Minimum bleach value",
    )
    parser.add_argument(
        "--bleach-max",
        type=int,
        default=1000,
        required=False,
        help="Maximum bleach value",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=None,
        required=False,
        help="Number/Fraction of cpus to use",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=600,
        required=False,
        help="Time budget (in seconds)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        required=False,
        help="Number of samples to run",
    )

    parser.add_argument(
        "--concurrent",
        type=int,
        default=None,
        required=False,
        help="Number of concurrent trials",
    )

    args = parser.parse_args()
    print(args)

    # Minimal preprocessing arguments
    root_data_dir = Path(args.root_data_dir).absolute()
    output_dir = Path(args.output_dir) / args.dataset / args.ram
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = output_dir.absolute()
    experiment_name = args.experiment_name or datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    resolution_min = args.resolution_min
    resolution_max = args.resolution_max
    assert resolution_min >= 4, "resolution_min must be >= 4"
    assert resolution_max <= 128, "resolution_max must be <= 128"
    assert (
        resolution_min <= resolution_max
    ), "resolution_min must be <= resolution_max"

    bleach_min = args.bleach_min
    bleach_max = args.bleach_max
    assert bleach_min >= 1, "bleach_min must be >= 1"
    assert bleach_min <= bleach_max, "bleach_min must be <= bleach_max"

    main(
        dataset=args.dataset,
        ram=args.ram,
        metric=args.metric,
        root_data_dir=root_data_dir,
        output_dir=output_dir,
        experiment_name=experiment_name,
        resolution_min=resolution_min,
        resolution_max=resolution_max,
        bleach_min=bleach_min,
        bleach_max=bleach_max,
        cpus=args.cpus,
        budget=args.budget,
        num_samples=args.samples,
        concurrent_trials=args.concurrent,
    )
