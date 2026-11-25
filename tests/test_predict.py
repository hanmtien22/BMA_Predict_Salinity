import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from importlib.metadata import version

import cloudpickle
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

import predict


class PredictionTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / "input.csv"
        self.output = self.root / "output.csv"
        self.frame = pd.DataFrame([
            dict.fromkeys(predict.FEATURES, 0.0), dict.fromkeys(predict.FEATURES, 1.0)
        ]).assign(date=["2020-01-01", "2020-02-29"], id=["001", "002"])
        self.frame.to_csv(self.source, index=False)
        base = DummyRegressor(strategy="constant", constant=-2).fit(np.zeros((1, 38)), [0])
        # Defines a known correction t + 2 sin + 3 cos + 4 x + 5 y.
        points = np.vstack((np.zeros(5), np.eye(5)))
        gp = LinearRegression().fit(points, points @ np.arange(1, 6))
        common = dict(schema_version=2, bundle_id="synthetic", gp_residual_kind="tree_only",
                      base_cols=predict.FEATURES + predict.TIME_FEATURES)
        state = dict(tree=base, gp=gp, blend_weights=(1., 0.))
        bundle = dict(common, store={predict.MODEL_DEFAULT: state}, preprocessing=dict(
            t_0="2020-01-01", lat0=0., lon0=0., xy_sd=[2., 3.]))
        data = cloudpickle.dumps(bundle)
        (self.root / "fitted_models.pkl").write_bytes(data)
        metadata = dict(common, fitted_models_sha256=hashlib.sha256(data).hexdigest(),
                        python_version=sys.version, package_versions={
                            p: version(p) for p in predict.RUNTIME_PACKAGES})
        (self.root / "parameters.json").write_text(json.dumps(metadata), encoding="utf-8")

    def run_prediction(self):
        return predict.predict_file(self.source, self.output, self.root, predict.MODEL_DEFAULT)

    def test_csv_round_trip(self):
        self.assertEqual(self.run_prediction(), (2, "synthetic"))
        output = pd.read_csv(self.output, dtype={"id": str})
        self.assertEqual(output.id.tolist(), ["001", "002"])
        angle = 2 * np.pi * np.array([1, 60]) / 365.25
        expected = (np.array([0, 59]) + 2*np.sin(angle) + 3*np.cos(angle)
                    + np.array([0, 4*111.32/2 + 5*110.57/3]))
        np.testing.assert_allclose(output[predict.PREDICTION], expected)

    def test_changed_pickle_is_rejected_before_loading(self):
        (self.root / "fitted_models.pkl").write_bytes(b"invalid pickle")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.run_prediction()
        self.assertFalse(self.output.exists())

    def test_invalid_data_preserves_previous_output(self):
        self.output.write_bytes(b"previous result")
        for field, value in [("date", "2020-02-30"), ("PCA4", np.inf), ("Lat", 91)]:
            with self.subTest(field=field):
                self.frame.assign(**{field: value}).to_csv(self.source, index=False)
                with self.assertRaises(ValueError):
                    self.run_prediction()
                self.assertEqual(self.output.read_bytes(), b"previous result")

    def test_missing_feature_is_named(self):
        self.frame.drop(columns="NDVI").to_csv(self.source, index=False)
        with self.assertRaisesRegex(ValueError, "NDVI"):
            self.run_prediction()
