"""Optional end-to-end replay using private files supplied by the model owner."""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

import predict


@unittest.skipUnless(os.environ.get("SALINITY_MODEL_DIR") and os.environ.get("SALINITY_SOURCE_CSV"),
                     "Set both private reference paths to replay the held-out observations.")
class ReferenceTest(unittest.TestCase):
    def test_all_exported_models(self):
        model_dir = Path(os.environ["SALINITY_MODEL_DIR"]).resolve()
        original = Path(os.environ["SALINITY_SOURCE_CSV"])
        meta = json.loads((model_dir / "parameters.json").read_text(encoding="utf-8"))
        self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(), meta["data_sha256"])
        source = pd.read_csv(original)
        flags = [c for c in source if re.fullmatch(r"\d{2}Th\d{4}", c)]
        active = source[flags].eq(1)
        source = source.loc[active.sum(axis=1) == 1].copy()
        dates = active.loc[source.index].idxmax(axis=1)
        source["date"] = pd.to_datetime(dates, format="%dTh%m%y").dt.strftime("%Y-%m-%d")
        held_out = source.loc[source.date >= meta["split_dates"]["train_end_exclusive"]]
        held_out = held_out.sort_values("date", kind="stable").reset_index(drop=True)
        reference = pd.read_csv(model_dir / "test_predictions.csv")
        self.assertEqual(set(reference.model), set(meta["gp_parameters"]))
        with tempfile.TemporaryDirectory() as tmp:
            observations = Path(tmp) / "input.csv"
            output = Path(tmp) / "output.csv"
            held_out[["date", *predict.FEATURES]].to_csv(observations, index=False)
            for name, rows in reference.groupby("model"):
                with self.subTest(model=name):
                    subprocess.run([sys.executable, predict.__file__, "--model-dir", str(model_dir),
                                    "--model", name, "--input", str(observations), "--output",
                                    str(output)], check=True)
                    actual = pd.read_csv(output)
                    self.assertEqual(actual.date.tolist(), rows.date.tolist())
                    np.testing.assert_allclose(actual[["Lat", "Long"]], rows[["Lat", "Long"]])
                    np.testing.assert_allclose(actual[predict.PREDICTION], rows.pred_hybrid,
                                               rtol=0, atol=1e-6)
