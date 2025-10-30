from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import ID_COLUMNS, PROCESSED_DIR, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE


@dataclass
class ProcessedPaths:
    X_train_path: Path
    X_test_path: Path
    y_train_path: Path
    y_test_path: Path
    preprocessor_path: Path


def find_csvs(raw_dir: Path) -> List[Path]:
    return sorted([p for p in raw_dir.glob("*.csv") if p.is_file()])


def load_raw_data(raw_dir: Path) -> pd.DataFrame:
    csvs = find_csvs(raw_dir)
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}. Download dataset first.")
    frames = [pd.read_csv(p) for p in csvs]
    df = pd.concat(frames, axis=0, ignore_index=True)
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop known ID columns if present
    for col in ID_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Normalize target column name and values if present in different cases
    if TARGET_COLUMN not in df.columns:
        # Try case-insensitive match
        matches = [c for c in df.columns if c.lower() == TARGET_COLUMN.lower()]
        if matches:
            df = df.rename(columns={matches[0]: TARGET_COLUMN})

    if TARGET_COLUMN in df.columns:
        # Map various churn encodings to 0/1
        mapping = {"Yes": 1, "No": 0, True: 1, False: 0, "True": 1, "False": 0, "1": 1, "0": 0}
        df[TARGET_COLUMN] = df[TARGET_COLUMN].map(mapping).fillna(df[TARGET_COLUMN]).astype(str)
        # After mapping, coerce numeric if possible
        try:
            df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN])
        except Exception:
            df[TARGET_COLUMN] = df[TARGET_COLUMN].replace({"Yes": 1, "No": 0}).map({1: 1, 0: 0}).fillna(0).astype(int)

    # Handle known Telco quirk: TotalCharges sometimes is string with spaces
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    return df


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    categorical_cols = [c for c in feature_cols if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
    numerical_cols = [c for c in feature_cols if c not in categorical_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor


def preprocess_and_split(
    raw_dir: Path,
    processed_dir: Path = PROCESSED_DIR,
) -> Tuple[ProcessedPaths, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw_data(raw_dir)
    df = clean_dataframe(df)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor(pd.concat([X_train, y_train], axis=1))

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    X_train_path = processed_dir / "X_train.parquet"
    X_test_path = processed_dir / "X_test.parquet"
    y_train_path = processed_dir / "y_train.parquet"
    y_test_path = processed_dir / "y_test.parquet"
    preprocessor_path = processed_dir / "preprocessor.joblib"

    # Save arrays and objects
    pd.DataFrame(X_train_processed).to_parquet(X_train_path)
    pd.DataFrame(X_test_processed).to_parquet(X_test_path)
    pd.Series(y_train).to_frame("y").to_parquet(y_train_path)
    pd.Series(y_test).to_frame("y").to_parquet(y_test_path)
    joblib.dump(preprocessor, preprocessor_path)

    paths = ProcessedPaths(
        X_train_path=X_train_path,
        X_test_path=X_test_path,
        y_train_path=y_train_path,
        y_test_path=y_test_path,
        preprocessor_path=preprocessor_path,
    )

    return paths, (X_train_processed, X_test_processed, y_train.to_numpy(), y_test.to_numpy())


