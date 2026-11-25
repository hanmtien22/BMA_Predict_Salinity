"""Batch salinity inference for the schema-v2 tree + GP bundle."""

import argparse
import csv
import hashlib
from importlib.metadata import version
from itertools import batched
import json
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

import cloudpickle
import numpy as np
import pandas as pd


FEATURES = (
    "DEM Lat Long EVI NDBI NDMI NDSI NDVI NDWI VSSI SAVI COSRI EVI2 MSI "
    "ND23 ND47 SI1 SI2 SI3 SI4 SI5 SI6 SI7 SI8 SI9 CA B G R NIR "
    "PCA1 PCA2 PCA3 PCA4 PCA5"
).split()
TIME_FEATURES = ["sin_doy", "cos_doy", "t"]
MODEL_DEFAULT = "PSO LightGBM"
PREDICTION = "salinity_pred_per_mille"
RUNTIME_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "cloudpickle",
                    "xgboost", "lightgbm")


def load_bundle(folder):
    metadata = json.loads((folder / "parameters.json").read_text(encoding="utf-8"))
    if metadata["schema_version"] != 2 or metadata["gp_residual_kind"] != "tree_only":
        raise ValueError("Expected a schema-v2 tree_only bundle.")
    expected_python = metadata["python_version"].split()[0].split(".")[:2]
    if list(map(int, expected_python)) != list(sys.version_info[:2]):
        raise ValueError(f"Use Python {'.'.join(expected_python)} for this bundle.")
    for package in RUNTIME_PACKAGES:
        required = metadata["package_versions"][package]
        if version(package) != required:
            raise ValueError(f"Install {package}=={required} for this bundle.")

    # Read once so the verified bytes are exactly the bytes deserialized.
    data = (folder / "fitted_models.pkl").read_bytes()
    if hashlib.sha256(data).hexdigest() != metadata["fitted_models_sha256"]:
        raise ValueError("Model SHA-256 differs from parameters.json.")
    bundle = cloudpickle.loads(data)
    for key in ("bundle_id", "schema_version", "gp_residual_kind", "base_cols"):
        if bundle[key] != metadata[key]:
            raise ValueError(f"Bundle and parameters.json disagree on {key}.")
    if bundle["base_cols"] != FEATURES + TIME_FEATURES:
        raise ValueError("Feature schema differs from the supported 38-column schema.")
    return bundle


def predict_rows(frame, bundle, model_name):
    state = bundle["store"][model_name]
    if tuple(state["blend_weights"]) != (1.0, 0.0):
        raise ValueError("Expected a tree-only base with blend weights (1, 0).")
    if not frame["date"].astype(str).str.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}").all():
        raise ValueError("Each date must be YYYY-MM-DD.")
    date = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise")
    features = frame[FEATURES].apply(pd.to_numeric, errors="raise").astype(float)
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("All 35 numeric features must be finite.")
    if not (features["Lat"].between(-90, 90).all() and features["Long"].between(-180, 180).all()):
        raise ValueError("Coordinates must be latitude and longitude in degrees.")

    prep = bundle["preprocessing"]
    angle = 2 * np.pi * date.dt.dayofyear / 365.25
    features["sin_doy"] = np.sin(angle)
    features["cos_doy"] = np.cos(angle)
    features["t"] = (date - pd.Timestamp(prep["t_0"])).dt.days
    x = (features["Long"] - prep["lon0"]) * 111.32 * np.cos(np.radians(prep["lat0"]))
    y = (features["Lat"] - prep["lat0"]) * 110.57
    gp_inputs = np.column_stack((features["t"], features["sin_doy"], features["cos_doy"],
                                 x / prep["xy_sd"][0], y / prep["xy_sd"][1]))
    tree = state["tree"]
    tree_inputs = features[bundle["base_cols"]]
    if hasattr(tree, "feature_names_in_"):
        # LightGBM assigned positional names when fitted on the notebook's ndarray.
        if list(tree.feature_names_in_) == [f"Column_{i}" for i in range(tree_inputs.shape[1])]:
            tree_inputs = tree_inputs.set_axis(tree.feature_names_in_, axis="columns")
    else:
        tree_inputs = tree_inputs.to_numpy()
    base = np.maximum(0, tree.predict(tree_inputs))
    result = np.maximum(0, base + state["gp"].predict(gp_inputs))
    if not np.isfinite(result).all():
        raise ValueError("Model produced a non-finite prediction.")
    return result


def read_batches(stream):
    reader = csv.reader(stream, strict=True)
    header = next(reader, [])
    missing = sorted(set(["date"] + FEATURES) - set(header))
    if missing:
        raise ValueError(f"Missing CSV columns: {', '.join(missing)}.")
    if len(header) != len(set(header)) or PREDICTION in header:
        raise ValueError("Duplicate header or existing prediction column.")
    for rows in batched(reader, 2048):
        if any(len(row) != len(header) for row in rows):
            raise ValueError("CSV row width differs from header.")
        yield pd.DataFrame.from_records(rows, columns=header)


def predict_file(source, destination, folder, model_name):
    if destination.suffix.lower() != ".csv" or source.resolve() == destination.resolve():
        raise ValueError("Select a separate output CSV.")

    bundle = load_bundle(folder)
    if model_name not in bundle["store"]:
        raise ValueError(f"Choose a model from: {', '.join(bundle['store'])}.")
    temp_path = None
    rows = 0
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", newline="", delete=False,
                                dir=destination.parent, prefix=".salinity-") as output:
            temp_path = Path(output.name)
            with source.open(encoding="utf-8-sig", newline="") as stream:
                for frame in read_batches(stream):
                    frame[PREDICTION] = predict_rows(frame, bundle, model_name)
                    frame.to_csv(output, index=False, header=(rows == 0))
                    rows += len(frame)
        if not rows:
            raise ValueError("CSV has no observations.")
        os.replace(temp_path, destination)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return rows, bundle["bundle_id"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Trusted local folder containing parameters.json and fitted_models.pkl")
    parser.add_argument("--model", default=MODEL_DEFAULT, help="Tree name in the bundle")
    args = parser.parse_args()
    try:
        rows, bundle_id = predict_file(args.input, args.output, args.model_dir, args.model)
    except (OSError, ValueError, KeyError, ImportError, csv.Error) as error:
        parser.exit(1, f"Prediction failed: {error}\n")
    print(f"{rows} rows | {args.model} + GP | bundle {bundle_id} | {args.output}")


if __name__ == "__main__":
    main()
