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

from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, f1_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from ray.tune.search import ConcurrencyLimiter

from wisardlib.builder import build_symmetric_wisard
from wisardlib.utils import untie_by_first_class
from sklearn.metrics import accuracy_score, f1_score
from time import perf_counter

from ray import tune
from ray.train import FailureConfig
import pickle

classifiers = {
    "random-forest": RandomForestClassifier,
    "knn": KNeighborsClassifier,
    "svm": SVC,
    "mlp": MLPClassifier,
    "mlp-2": MLPClassifier,
}


@dataclass
class ClassifierConfig:
    name: str
    params: dict = field(default_factory=dict)
    
def get_model_size(model):
    return len(pickle.dumps(model))


datasets = {
    "iris_fold_0": {"path": Path("iris_fold_0/data.pkl")},
    "iris_fold_1": {"path": Path("iris_fold_1/data.pkl")},
    "iris_fold_2": {"path": Path("iris_fold_2/data.pkl")},
    "iris_fold_3": {"path": Path("iris_fold_3/data.pkl")},
    "iris_fold_4": {"path": Path("iris_fold_4/data.pkl")},
    
    "wine_fold_0": {"path": Path("wine_fold_0/data.pkl")},
    "wine_fold_1": {"path": Path("wine_fold_1/data.pkl")},
    "wine_fold_2": {"path": Path("wine_fold_2/data.pkl")},
    "wine_fold_3": {"path": Path("wine_fold_3/data.pkl")},
    "wine_fold_4": {"path": Path("wine_fold_4/data.pkl")},
    
    "breast_cancer_fold_0": {"path": Path("breast_cancer_fold_0/data.pkl")},
    "breast_cancer_fold_1": {"path": Path("breast_cancer_fold_1/data.pkl")},
    "breast_cancer_fold_2": {"path": Path("breast_cancer_fold_2/data.pkl")},
    "breast_cancer_fold_3": {"path": Path("breast_cancer_fold_3/data.pkl")},
    "breast_cancer_fold_4": {"path": Path("breast_cancer_fold_4/data.pkl")},
    
    "ecoli_fold_0": {"path": Path("ecoli_fold_0/data.pkl")},
    "ecoli_fold_1": {"path": Path("ecoli_fold_1/data.pkl")},
    "ecoli_fold_2": {"path": Path("ecoli_fold_2/data.pkl")},
    "ecoli_fold_3": {"path": Path("ecoli_fold_3/data.pkl")},
    "ecoli_fold_4": {"path": Path("ecoli_fold_4/data.pkl")},
    
    "letter_fold_0": {"path": Path("letter_fold_0/data.pkl")},
    "letter_fold_1": {"path": Path("letter_fold_1/data.pkl")},
    "letter_fold_2": {"path": Path("letter_fold_2/data.pkl")},
    "letter_fold_3": {"path": Path("letter_fold_3/data.pkl")},
    "letter_fold_4": {"path": Path("letter_fold_4/data.pkl")},
    
    "satimage_fold_0": {"path": Path("satimage_fold_0/data.pkl")},
    "satimage_fold_1": {"path": Path("satimage_fold_1/data.pkl")},
    "satimage_fold_2": {"path": Path("satimage_fold_2/data.pkl")},
    "satimage_fold_3": {"path": Path("satimage_fold_3/data.pkl")},
    "satimage_fold_4": {"path": Path("satimage_fold_4/data.pkl")},
    
    "segment_fold_0": {"path": Path("segment_fold_0/data.pkl")},
    "segment_fold_1": {"path": Path("segment_fold_1/data.pkl")},
    "segment_fold_2": {"path": Path("segment_fold_2/data.pkl")},
    "segment_fold_3": {"path": Path("segment_fold_3/data.pkl")},
    "segment_fold_4": {"path": Path("segment_fold_4/data.pkl")},
    
    "glass_fold_0": {"path": Path("glass_fold_0/data.pkl")},
    "glass_fold_1": {"path": Path("glass_fold_1/data.pkl")},
    "glass_fold_2": {"path": Path("glass_fold_2/data.pkl")},
    "glass_fold_3": {"path": Path("glass_fold_3/data.pkl")},
    "glass_fold_4": {"path": Path("glass_fold_4/data.pkl")},
    
    "vehicle_fold_0": {"path": Path("vehicle_fold_0/data.pkl")},
    "vehicle_fold_1": {"path": Path("vehicle_fold_1/data.pkl")},
    "vehicle_fold_2": {"path": Path("vehicle_fold_2/data.pkl")},
    "vehicle_fold_3": {"path": Path("vehicle_fold_3/data.pkl")},
    "vehicle_fold_4": {"path": Path("vehicle_fold_4/data.pkl")},
    
    "motion_sense_fold_0": {"path": Path("motion_sense_fold_0/data.pkl")},
    "motion_sense_fold_1": {"path": Path("motion_sense_fold_1/data.pkl")},
    "motion_sense_fold_2": {"path": Path("motion_sense_fold_2/data.pkl")},
    "motion_sense_fold_3": {"path": Path("motion_sense_fold_3/data.pkl")},
    "motion_sense_fold_4": {"path": Path("motion_sense_fold_4/data.pkl")},
    
    "optical_handwritten_fold_0": {"path": Path("optical_handwritten_fold_0/data.pkl")},
    "optical_handwritten_fold_1": {"path": Path("optical_handwritten_fold_1/data.pkl")},
    "optical_handwritten_fold_2": {"path": Path("optical_handwritten_fold_2/data.pkl")},
    "optical_handwritten_fold_3": {"path": Path("optical_handwritten_fold_3/data.pkl")},
    "optical_handwritten_fold_4": {"path": Path("optical_handwritten_fold_4/data.pkl")},
    
    "image_segmentation_fold_0": {"path": Path("image_segmentation_fold_0/data.pkl")},
    "image_segmentation_fold_1": {"path": Path("image_segmentation_fold_1/data.pkl")},
    "image_segmentation_fold_2": {"path": Path("image_segmentation_fold_2/data.pkl")},
    "image_segmentation_fold_3": {"path": Path("image_segmentation_fold_3/data.pkl")},
    "image_segmentation_fold_4": {"path": Path("image_segmentation_fold_4/data.pkl")},
    
    "sepsis_fold_0": {"path": Path("sepsis_fold_0/data.pkl")},
    "sepsis_fold_1": {"path": Path("sepsis_fold_1/data.pkl")},
    "sepsis_fold_2": {"path": Path("sepsis_fold_2/data.pkl")},
    "sepsis_fold_3": {"path": Path("sepsis_fold_3/data.pkl")},
    "sepsis_fold_4": {"path": Path("sepsis_fold_4/data.pkl")},
    
    "rice_fold_0": {"path": Path("rice_fold_0/data.pkl")},
    "rice_fold_1": {"path": Path("rice_fold_1/data.pkl")},
    "rice_fold_2": {"path": Path("rice_fold_2/data.pkl")},
    "rice_fold_3": {"path": Path("rice_fold_3/data.pkl")},
    "rice_fold_4": {"path": Path("rice_fold_4/data.pkl")},
    
    "yeast_fold_0": {"path": Path("yeast_fold_0/data.pkl")},
    "yeast_fold_1": {"path": Path("yeast_fold_1/data.pkl")},
    "yeast_fold_2": {"path": Path("yeast_fold_2/data.pkl")},
    "yeast_fold_3": {"path": Path("yeast_fold_3/data.pkl")},
    "yeast_fold_4": {"path": Path("yeast_fold_4/data.pkl")},
    
    "dry_bean_fold_0": {"path": Path("dry_bean_fold_0/data.pkl")},
    "dry_bean_fold_1": {"path": Path("dry_bean_fold_1/data.pkl")},
    "dry_bean_fold_2": {"path": Path("dry_bean_fold_2/data.pkl")},
    "dry_bean_fold_3": {"path": Path("dry_bean_fold_3/data.pkl")},
    "dry_bean_fold_4": {"path": Path("dry_bean_fold_4/data.pkl")},
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



class BaseExperiment(tune.Trainable):
    MODEL_name = ""

    @staticmethod
    def get_search_space():
        raise NotImplementedError

    def _get_ram_config(self, config: Dict) -> Dict:
        raise NotImplementedError

    def _get_metrics(self, metrics, y_pred, y_true, stage: str):
        metrics["accuracy"].append(
            accuracy_score(y_true, y_pred),
        )
        metrics["f1 weighted"].append(
            f1_score(y_true, y_pred, average="weighted")
        )     
        metrics["size"].append(get_model_size(self.model))

        result = {
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

    # vars:
    # root_data_dir: Path
    # dataset_name: str

    def setup(self, config):
        self.val_metrics = defaultdict(list)
        self.test_metrics = defaultdict(list)

        self.root_data_dir = Path(config["root_data_dir"])
        self.dataset_name = config["dataset_name"]
        self.model_config = self._get_config(config)

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

        # Transform data
        self.x_train = self.x_train.reshape(self.x_train.shape[0], -1)
        self.x_val = self.x_val.reshape(self.x_val.shape[0], -1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], -1)
        # Discretize labels
        self.y_train = discretize_labels(self.y_train)
        self.y_val = discretize_labels(self.y_val)
        self.y_test = discretize_labels(self.y_test)
        # Indices and classes
        self.n_classes = max(self.y_train) + 1
        

    def step(self):
        self.model_cls = classifiers[self.MODEL_name]
        self.model = self.model_cls(**self.model_config)
        
        with Timer() as t_train_time:
            self.model.fit(self.x_train, self.y_train)

        with Timer() as t_val_predict_time:
            y_pred = self.model.predict(self.x_val)
            
        metrics = self._get_metrics(
            metrics=self.val_metrics,
            y_pred=y_pred,
            y_true=self.y_val,
            stage="val",
        )
        
        metrics["train time"] = t_train_time.time
        metrics["val_predict time"] = t_val_predict_time.time
        metrics["train_samples"] = len(self.x_train)
        metrics["val_samples"] = len(self.x_val)
        metrics["test_samples"] = len(self.x_test)
        metrics["classes"] = self.n_classes
        
        
        with Timer() as t_test_predict_time:
            y_pred = self.model.predict(self.x_test)
            
        test_metrics = self._get_metrics(
            metrics=self.test_metrics,
            y_pred=y_pred,
            y_true=self.y_test,
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


class RandomForestExperiment(BaseExperiment):
    MODEL_name = "random-forest"
    
    def _get_config(self, config: Dict) -> Dict:
        return {
            "n_estimators": config["n_estimators"],
            "max_depth": config["max_depth"],
        }
    
    @staticmethod
    def get_search_space():
        return {
            "n_estimators": optuna.distributions.IntDistribution(10, 1000),
            "max_depth": optuna.distributions.IntDistribution(1, 100),
        }

class KNNExperiment(BaseExperiment):
    MODEL_name = "knn"
    
    @staticmethod
    def get_search_space():
        return {
            "n_neighbors": optuna.distributions.IntDistribution(1, 100),
            "weights": optuna.distributions.CategoricalDistribution(["uniform", "distance"]),
            "algorithm": optuna.distributions.CategoricalDistribution(["auto", "ball_tree", "kd_tree", "brute"]),
        }
        
    def _get_config(self, config: Dict) -> Dict:
        return {
            "n_neighbors": config["n_neighbors"],
            "weights": config["weights"],
            "algorithm": config["algorithm"],
        }
    
    
class SVMExperiment(BaseExperiment):
    MODEL_name = "svm"
    
    @staticmethod
    def get_search_space():
        return {
            "C": optuna.distributions.FloatDistribution(0.1, 1000),
            "kernel": optuna.distributions.CategoricalDistribution(["linear", "poly", "rbf", "sigmoid"]),
            "degree": optuna.distributions.IntDistribution(1, 10),
            "gamma": optuna.distributions.CategoricalDistribution(["scale", "auto"]),
        }
        
    def _get_config(self, config: Dict) -> Dict:
        return {
            "C": config["C"],
            "kernel": config["kernel"],
            "degree": config["degree"],
            "gamma": config["gamma"],
        }

class MLPExperiment(BaseExperiment):
    MODEL_name = "mlp-2"
    
    @staticmethod
    def get_search_space():
        return {
            "hidden_layer_sizes": optuna.distributions.IntDistribution(1, 300),
            "activation": optuna.distributions.CategoricalDistribution(["logistic", "tanh", "relu"]),
            "solver": optuna.distributions.CategoricalDistribution(["lbfgs", "sgd", "adam"]),
            "alpha": optuna.distributions.FloatDistribution(0.0001, 0.1),
            "learning_rate": optuna.distributions.CategoricalDistribution(["constant", "invscaling", "adaptive"]),
        }
        
    def _get_config(self, config: Dict) -> Dict:
        return {
            "hidden_layer_sizes": (config["hidden_layer_sizes"], ),
            "activation": config["activation"],
            "solver": config["solver"],
            "alpha": config["alpha"],
            "learning_rate": config["learning_rate"],
        }


class MLP2Experiment(BaseExperiment):
    MODEL_name = "mlp"
    
    @staticmethod
    def get_search_space():
        return {
            "hidden_layer_sizes_1": optuna.distributions.IntDistribution(1, 300),
            "hidden_layer_sizes_2": optuna.distributions.IntDistribution(1, 300),
            "activation": optuna.distributions.CategoricalDistribution(["logistic", "tanh", "relu"]),
            "solver": optuna.distributions.CategoricalDistribution(["lbfgs", "sgd", "adam"]),
            "alpha": optuna.distributions.FloatDistribution(0.0001, 0.1),
            "learning_rate": optuna.distributions.CategoricalDistribution(["constant", "invscaling", "adaptive"]),
        }
        
    def _get_config(self, config: Dict) -> Dict:
        return {
            "hidden_layer_sizes": (config["hidden_layer_sizes_1"], config["hidden_layer_sizes_2"]),
            "activation": config["activation"],
            "solver": config["solver"],
            "alpha": config["alpha"],
            "learning_rate": config["learning_rate"],
        }

    

def main(
    dataset: str,
    model: str,
    root_data_dir: Path,
    metric: str,
    output_dir: Path,
    experiment_name: str,
    cpus: int,
    budget: int,
    num_samples: int = 10000,
    concurrent_trials: int = None,
):
    experiment_map = {
        "random-forest": RandomForestExperiment,
        "knn": KNNExperiment,
        "svm": SVMExperiment,
        "mlp": MLPExperiment,
        "mlp-2": MLP2Experiment,
    }

    experiment_cls = experiment_map[model]

    # ------  Search space -------
    space = {
        "root_data_dir": optuna.distributions.CategoricalDistribution(
            [str(root_data_dir)]
        ),
        "dataset_name": optuna.distributions.CategoricalDistribution([dataset]),
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
        "--model",
        choices=list(classifiers.keys()),
        type=str,
        required=True,
        help="Classifier to use",
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
    output_dir = Path(args.output_dir) / args.dataset / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = output_dir.absolute()
    experiment_name = args.experiment_name or datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )


    main(
        dataset=args.dataset,
        model=args.model,
        root_data_dir=root_data_dir,
        metric=args.metric,
        output_dir=output_dir,
        experiment_name=experiment_name,
        cpus=args.cpus,
        budget=args.budget,
        num_samples=args.samples,
        concurrent_trials=args.concurrent,
    )
