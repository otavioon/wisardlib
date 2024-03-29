from wisardlib.encoders.thermometer import (
    ThermometerEncoder,
    DistributiveThermometerEncoder,
)
from pathlib import Path
import numpy as np
import pandas as pd
import random
import logging
import traceback
import argparse
import time

from base_runner import run_experiment, RAMConfig, datasets, encoders

configs = [
    # Dict RAM
    RAMConfig(name="DictRam", RAM_cls_name="dict"),
    # Bloom filter
    # RAMConfig(
    #     name="count-bloom",
    #     RAM_cls_name="count-bloom",
    #     RAM_cls_kwargs={"est_elements": 1000, "false_positive_rate": 0.05},
    # ),
    # RAMConfig(
    #     name="count-bloom",
    #     RAM_cls_name="count-bloom",
    #     RAM_cls_kwargs={"est_elements": 100, "false_positive_rate": 0.05},
    # ),
    # RAMConfig(
    #     name="count-bloom",
    #     RAM_cls_name="count-bloom",
    #     RAM_cls_kwargs={"est_elements": 100, "false_positive_rate": 0.02},
    # ),
    # RAMConfig(
    #     name="count-bloom",
    #     RAM_cls_name="count-bloom",
    #     RAM_cls_kwargs={"est_elements": 100, "false_positive_rate": 0.08},
    # ),
    # # Count-Min-Sketch
    # RAMConfig(
    #     name="count-min-sketch",
    #     RAM_cls_name="count-min-sketch",
    #     RAM_cls_kwargs={"width": 1000, "depth": 5},
    # ),
    # RAMConfig(
    #     name="count-min-sketch",
    #     RAM_cls_name="count-min-sketch",
    #     RAM_cls_kwargs={"width": 500, "depth": 5},
    # ),
    # RAMConfig(
    #     name="count-min-sketch",
    #     RAM_cls_name="count-min-sketch",
    #     RAM_cls_kwargs={"width": 100, "depth": 5},
    # ),
    # RAMConfig(
    #     name="count-min-sketch",
    #     RAM_cls_name="count-min-sketch",
    #     RAM_cls_kwargs={"width": 1000, "depth": 3},
    # ),
    # RAMConfig(
    #     name="count-min-sketch",
    #     RAM_cls_name="count-min-sketch",
    #     RAM_cls_kwargs={"width": 500, "depth": 3},
    # ),
    # RAMConfig(
    #     name="count-min-sketch",
    #     RAM_cls_name="count-min-sketch",
    #     RAM_cls_kwargs={"width": 100, "depth": 3},
    # ),
    # RAMConfig(
    #     name="count-min-sketch",
    #     RAM_cls_name="count-min-sketch",
    #     RAM_cls_kwargs={"width": 50, "depth": 3},
    # ),
    # RAMConfig(
    #     name="count-min-sketch",
    #     RAM_cls_name="count-min-sketch",
    #     RAM_cls_kwargs={"width": 50, "depth": 2},
    # ),
    # RAMConfig(
    #     name="count-min-sketch",
    #     RAM_cls_name="count-min-sketch",
    #     RAM_cls_kwargs={"width": 20, "depth": 3},
    # ),
    # RAMConfig(
    #     name="count-min-sketch",
    #     RAM_cls_name="count-min-sketch",
    #     RAM_cls_kwargs={"width": 20, "depth": 2},
    # ),
    # # Count cuckoo
    # RAMConfig(
    #     name="count-cuckoo",
    #     RAM_cls_name="count-cuckoo",
    #     RAM_cls_kwargs={"capacity": 1000, "bucket_size": 4},
    # ),
    # RAMConfig(
    #     name="count-cuckoo",
    #     RAM_cls_name="count-cuckoo",
    #     RAM_cls_kwargs={"capacity": 500, "bucket_size": 4},
    # ),
    # RAMConfig(
    #     name="count-cuckoo",
    #     RAM_cls_name="count-cuckoo",
    #     RAM_cls_kwargs={"capacity": 100, "bucket_size": 4},
    # ),
    # RAMConfig(
    #     name="count-cuckoo",
    #     RAM_cls_name="count-cuckoo",
    #     RAM_cls_kwargs={"capacity": 1000, "bucket_size": 2},
    # ),
    # RAMConfig(
    #     name="count-cuckoo",
    #     RAM_cls_name="count-cuckoo",
    #     RAM_cls_kwargs={"capacity": 500, "bucket_size": 2},
    # ),
    # RAMConfig(
    #     name="count-cuckoo",
    #     RAM_cls_name="count-cuckoo",
    #     RAM_cls_kwargs={"capacity": 100, "bucket_size": 2},
    # ),
    # # Heavy Hitters
    # RAMConfig(
    #     name="heavy-hitters",
    #     RAM_cls_name="heavy-hitters",
    #     RAM_cls_kwargs={"num_hitters": 100, "width": 1000, "depth": 5},
    # ),
    # RAMConfig(
    #     name="heavy-hitters",
    #     RAM_cls_name="heavy-hitters",
    #     RAM_cls_kwargs={"num_hitters": 100, "width": 1000, "depth": 3},
    # ),
    # RAMConfig(
    #     name="heavy-hitters",
    #     RAM_cls_name="heavy-hitters",
    #     RAM_cls_kwargs={"num_hitters": 50, "width": 100, "depth": 5},
    # ),
    # RAMConfig(
    #     name="heavy-hitters",
    #     RAM_cls_name="heavy-hitters",
    #     RAM_cls_kwargs={"num_hitters": 50, "width": 100, "depth": 3},
    # ),
    # RAMConfig(
    #     name="heavy-hitters",
    #     RAM_cls_name="heavy-hitters",
    #     RAM_cls_kwargs={"num_hitters": 50, "width": 20, "depth": 3},
    # ),
    # RAMConfig(
    #     name="heavy-hitters",
    #     RAM_cls_name="heavy-hitters",
    #     RAM_cls_kwargs={"num_hitters": 50, "width": 20, "depth": 2},
    # ),
    # RAMConfig(
    #     name="heavy-hitters",
    #     RAM_cls_name="heavy-hitters",
    #     RAM_cls_kwargs={"num_hitters": 10, "width": 50, "depth": 3},
    # ),
    # RAMConfig(
    #     name="heavy-hitters",
    #     RAM_cls_name="heavy-hitters",
    #     RAM_cls_kwargs={"num_hitters": 10, "width": 50, "depth": 2},
    # ),
    # # Stream Threshold
    # RAMConfig(
    #     name="stream-threshold",
    #     RAM_cls_name="stream-threshold",
    #     RAM_cls_kwargs={"threshold": 100, "width": 1000, "depth": 5},
    # ),
    # RAMConfig(
    #     name="stream-threshold",
    #     RAM_cls_name="stream-threshold",
    #     RAM_cls_kwargs={"threshold": 50, "width": 1000, "depth": 5},
    # ),
    # RAMConfig(
    #     name="stream-threshold",
    #     RAM_cls_name="stream-threshold",
    #     RAM_cls_kwargs={"threshold": 100, "width": 250, "depth": 5},
    # ),
    # RAMConfig(
    #     name="stream-threshold",
    #     RAM_cls_name="stream-threshold",
    #     RAM_cls_kwargs={"threshold": 50, "width": 250, "depth": 5},
    # ),
    # RAMConfig(
    #     name="stream-threshold",
    #     RAM_cls_name="stream-threshold",
    #     RAM_cls_kwargs={"threshold": 100, "width": 1000, "depth": 3},
    # ),
    # RAMConfig(
    #     name="stream-threshold",
    #     RAM_cls_name="stream-threshold",
    #     RAM_cls_kwargs={"threshold": 50, "width": 1000, "depth": 3},
    # ),
    # RAMConfig(
    #     name="stream-threshold",
    #     RAM_cls_name="stream-threshold",
    #     RAM_cls_kwargs={"threshold": 100, "width": 250, "depth": 3},
    # ),
    # RAMConfig(
    #     name="stream-threshold",
    #     RAM_cls_name="stream-threshold",
    #     RAM_cls_kwargs={"threshold": 50, "width": 250, "depth": 3},
    # ),
    # RAMConfig(
    #     name="stream-threshold",
    #     RAM_cls_name="stream-threshold",
    #     RAM_cls_kwargs={"threshold": 50, "width": 100, "depth": 3},
    # ),
    # RAMConfig(
    #     name="stream-threshold",
    #     RAM_cls_name="stream-threshold",
    #     RAM_cls_kwargs={"threshold": 50, "width": 100, "depth": 2},
    # ),
]


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
        "-r",
        "--runs",
        action="store",
        help="Number of runs",
        default=3,
        type=int,
        required=False,
    )
    parser.add_argument(
        "-b",
        "--bleach",
        action="store",
        nargs="+",
        help="Bleach values",
        default=[3],
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

    args = parser.parse_args()
    print(args)

    formatter = "[%(asctime)s] [%(levelname)s]: %(message)s"
    logging.basicConfig(format=formatter, level=args.level)

    start_time = time.time()
    run_experiment(
        exp_name=args.experiment_name,
        output_path=args.output_path,
        dataset_name=args.dataset,
        ram_configs=configs,
        encoder_name=args.encoder,
        encoder_kwargs=dict(resolution=args.resolution),
        tuple_size=args.tuple_size,
        number_runs=args.runs,
        bleach=args.bleach,
        root_data_path=args.root_data_dir,
    )
    end_time = time.time()

    print(f"Done! It took {end_time-start_time:.3f} seconds!")
