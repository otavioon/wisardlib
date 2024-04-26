from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pickle

original_data_path = Path("/workspaces/wisard/experiments/data")
fold_path = Path(".")


for dataset in original_data_path.iterdir():
    if not dataset.is_dir():
        continue

    dataset_name = dataset.name
    print(f"Processing dataset {dataset_name}")

    (x_train, y_train), (x_test, y_test) = np.load(
        dataset / "data.pkl", allow_pickle=True
    )

    X, y = np.concatenate([x_train, x_test]), np.concatenate([y_train, y_test])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold_no, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        dataset_fold_path = fold_path / f"{dataset_name}_fold_{fold_no}"
        dataset_fold_path.mkdir(exist_ok=True, parents=True)
        data_path = dataset_fold_path / "data.pkl"

        with open(data_path, "wb") as f:
            pickle.dump(((X_train, y_train), (X_test, y_test)), f)

        print(f"Fold {fold_no} saved at {data_path}")
