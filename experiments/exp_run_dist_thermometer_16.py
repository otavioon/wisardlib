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

from base import run_experiment, RAMConfig

formatter = "[%(asctime)s] [%(levelname)s]: %(message)s"
logging.basicConfig(format=formatter, level=logging.INFO)


configs = [
    RAMConfig(name="DictRam", RAM_cls_name="dict"),
    # Bloom 1
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
    RAMConfig(
        name="count-cuckoo",
        RAM_cls_name="count-cuckoo",
        RAM_cls_kwargs={"capacity": 1000, "bucket_size": 4},
    ),
    RAMConfig(
        name="heavy-hitters",
        RAM_cls_name="heavy-hitters",
        RAM_cls_kwargs={"num_hitters": 100, "width": 1000, "depth": 5},
    ),
    RAMConfig(
        name="stream-threshold",
        RAM_cls_name="stream-threshold",
        RAM_cls_kwargs={"threshold": 100, "width": 1000, "depth": 5},
    ),
    # Bloom 2
    RAMConfig(
        name="count-bloom",
        RAM_cls_name="count-bloom",
        RAM_cls_kwargs={"est_elements": 100, "false_positive_rate": 0.05},
    ),
    RAMConfig(
        name="count-min-sketch",
        RAM_cls_name="count-min-sketch",
        RAM_cls_kwargs={"width": 100, "depth": 5},
    ),
    RAMConfig(
        name="count-cuckoo",
        RAM_cls_name="count-cuckoo",
        RAM_cls_kwargs={"capacity": 100, "bucket_size": 4},
    ),
    RAMConfig(
        name="heavy-hitters",
        RAM_cls_name="heavy-hitters",
        RAM_cls_kwargs={"num_hitters": 50, "width": 100, "depth": 5},
    ),
    RAMConfig(
        name="stream-threshold",
        RAM_cls_name="stream-threshold",
        RAM_cls_kwargs={"threshold": 50, "width": 1000, "depth": 5},
    ),
]

# try:
#     df = run_experiment(
#         exp_name="IRIS Exp",
#         output_path="experiments/results",
#         ram_configs=configs,
#         dataset_name="iris",
#         encoder_name="distributive-thermometer",
#         encoder_kwargs={"resolution": 16},
#         tuple_size=16,
#         number_runs=3,
#         bleach=[2,5]
#     )
# except Exception as e:
#     print(f"Error (exp1): {e}")
#
#
# try:
#     df = run_experiment(
#         exp_name="Wine Exp",
#         output_path="experiments/results",
#         ram_configs=configs,
#         dataset_name="wine",
#         encoder_name="distributive-thermometer",
#         encoder_kwargs={"resolution": 16},
#         tuple_size=16,
#         number_runs=3,
#         bleach=[2,5]
#     )
# except Exception as e:
#     print(f"Error (exp2): {e}")
#
try:
    df = run_experiment(
        exp_name="breast_cancer Exp",
        output_path="experiments/results",
        ram_configs=configs,
        dataset_name="breast_cancer",
        encoder_name="distributive-thermometer",
        encoder_kwargs={"resolution": 16},
        tuple_size=16,
        number_runs=3,
        bleach=[2, 5],
    )
except Exception as e:
    print(f"Error (exp3): {e}")

# try:
#     df = run_experiment(
#         exp_name="ecoli Exp",
#         output_path="experiments/results",
#         ram_configs=configs,
#         dataset_name="ecoli",
#         encoder_name="distributive-thermometer",
#         encoder_kwargs={"resolution": 16},
#         tuple_size=16,
#         number_runs=3,
#         bleach=[2,5]
#     )
# except Exception as e:
#     print(f"Error (exp4): {e}")
#
# try:
#     df = run_experiment(
#         exp_name="letter Exp",
#         output_path="experiments/results",
#         ram_configs=configs,
#         dataset_name="letter",
#         encoder_name="distributive-thermometer",
#         encoder_kwargs={"resolution": 16},
#         tuple_size=16,
#         number_runs=3,
#         bleach=[2,5]
#     )
# except Exception as e:
#     print(f"Error (exp5): {e}")
#
# try:
#     df = run_experiment(
#         exp_name="satimage Exp",
#         output_path="experiments/results",
#         ram_configs=configs,
#         dataset_name="satimage",
#         encoder_name="distributive-thermometer",
#         encoder_kwargs={"resolution": 16},
#         tuple_size=16,
#         number_runs=3,
#         bleach=[2,5]
#     )
# except Exception as e:
#     print(f"Error (exp6): {e}")
#
# try:
#     df = run_experiment(
#         exp_name="segment Exp",
#         output_path="experiments/results",
#         ram_configs=configs,
#         dataset_name="segment",
#         encoder_name="distributive-thermometer",
#         encoder_kwargs={"resolution": 16},
#         tuple_size=16,
#         number_runs=3,
#         bleach=[2,5]
#     )
# except Exception as e:
#     print(traceback.format_exc())
#
# try:
#     df = run_experiment(
#         exp_name="glass Exp",
#         output_path="experiments/results",
#         ram_configs=configs,
#         dataset_name="glass",
#         encoder_name="distributive-thermometer",
#         encoder_kwargs={"resolution": 16},
#         tuple_size=16,
#         number_runs=3,
#         bleach=[2,5]
#     )
# except Exception as e:
#     print(f"Error (exp8): {e}")
