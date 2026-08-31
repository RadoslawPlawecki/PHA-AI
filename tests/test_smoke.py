"""Stage-2 sanity: the package imports and the test infrastructure loads."""

import numpy as np

from tests._helpers.assertions import assert_feature_matrix
from tests._helpers.fakes import FakeEstimator, FakeValidator, patch_prompts
from tests._helpers.frames import contig_feature_df, extracted_matrix


def test_import_phantom():
    import phantom

    assert phantom.__name__ == "phantom"


def test_key_modules_import():
    from phantom.classification.data.labeling import Labeling
    from phantom.classification.ml.evaluator import EvaluatorSl
    from phantom.config.features import FeatureConfigManager
    from phantom.features.pipelines.matrix import build_matrix

    assert all(callable(x) for x in (Labeling.derive_label, EvaluatorSl.evaluate, build_matrix))
    assert FeatureConfigManager is not None


def test_frames_builder():
    df = contig_feature_df([("geN|S1_c1", "A"), ("geN|S2_c1", "B")])
    assert list(df.columns) == ["Accession", "taxon"]
    assert len(df) == 2


def test_fake_estimator_is_deterministic():
    X = np.array([[0.1], [0.9], [0.2], [0.8]])
    y = np.array([0, 1, 0, 1])
    model = FakeEstimator().fit(X, y)
    assert model.predict(X).tolist() == [0, 1, 0, 1]
    assert model.predict_proba(X).shape == (4, 2)
    assert len(model.feature_importances_) == 1


def test_fake_validator_returns_cvresults():
    X = np.array([[0.1], [0.9], [0.2], [0.8]])
    y = np.array([0, 1, 0, 1])
    res = FakeValidator().run(FakeEstimator(), X, y)
    assert res.y_true.tolist() == [0, 1, 0, 1]
    assert res.test_idx.tolist() == [0, 1, 2, 3]


def test_patch_prompts_helper(monkeypatch):
    class _P:
        @staticmethod
        def ask():
            return "real"

    patch_prompts(monkeypatch, _P, {"ask": iter(["a", "b"])})
    assert [_P.ask(), _P.ask()] == ["a", "b"]


def test_assert_feature_matrix_accepts_valid():
    m = extracted_matrix([("geN|S1", 0.1, 0.9), ("geN|S2", 0.5, 0.5)], ["f1", "f2"])
    assert_feature_matrix(m)


def test_tmp_config_fixture(tmp_config):
    text = tmp_config.read_text()
    assert "[features.v9_9]" in text
    assert tmp_config.name == "config.toml"


def test_phantom_data_root_redirects_loader(phantom_data_root):
    from phantom.config.loader import ConfigLoader

    assert phantom_data_root == ConfigLoader.DATA_ROOT
    assert ConfigLoader.resolve_data_path("data/x") == phantom_data_root / "data/x"


def test_tmp_workspace_layout(tmp_workspace):
    assert tmp_workspace.config_path.exists()
    assert tmp_workspace.stage_dir("raw_merged").is_dir()
    assert (tmp_workspace.data_root / "masks" / "genomad").is_dir()
