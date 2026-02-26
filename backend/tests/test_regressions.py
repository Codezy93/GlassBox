import numpy as np

from backend.app import allowed_origins
from backend.data.preprocess import ALL_FEATURES, CONTINUOUS_FEATURES
from backend.engine.counterfactual import CounterfactualEngine
from backend.engine.formal_verifier import FormalVerifier


class _ProbeModel:
    def __init__(self):
        self.last_df = None

    def predict_proba(self, df):
        self.last_df = df.copy()
        return np.array([[0.62, 0.38]])

    def predict(self, df):
        self.last_df = df.copy()
        return np.array([0])


class _VerifierModel:
    def predict_proba(self, df):
        # Approval probability is mostly driven by pay_sep in this test model.
        pay_sep = df["pay_sep"].to_numpy(dtype=float)
        p0 = np.clip(0.45 + 0.30 * pay_sep, 0.0, 1.0)
        p1 = 1.0 - p0
        return np.column_stack([p0, p1])


def _sample_profile():
    profile = {name: 0.0 for name in ALL_FEATURES}
    profile["sex"] = 1
    profile["education"] = 2
    profile["marriage"] = 1
    profile["age"] = 35
    return profile


def test_counterfactual_predict_uses_raw_feature_space():
    engine = CounterfactualEngine.__new__(CounterfactualEngine)
    engine.ebm = _ProbeModel()

    profile = _sample_profile()
    profile["credit_limit"] = 120000.0
    result = engine.predict(profile)

    assert result["prediction"] == 0
    assert np.isclose(result["probability_no_default"], 0.62)
    assert engine.ebm.last_df is not None
    assert np.isclose(float(engine.ebm.last_df.iloc[0]["credit_limit"]), 120000.0)


def test_formal_verifier_is_deterministic_for_identical_input():
    verifier = FormalVerifier.__new__(FormalVerifier)
    verifier.model = _VerifierModel()
    verifier.scaler = None
    verifier.feature_scales = np.ones(len(CONTINUOUS_FEATURES), dtype=float)

    profile = _sample_profile()

    a = verifier.certify_recourse(profile, delta=0.2)
    b = verifier.certify_recourse(profile, delta=0.2)

    assert a["certified"] is True
    assert b["certified"] is True
    assert a["proof_id"] == b["proof_id"]


def test_default_cors_origins_are_explicit_localhosts():
    assert isinstance(allowed_origins, list)
    assert "*" not in allowed_origins
    assert "http://localhost:5173" in allowed_origins
