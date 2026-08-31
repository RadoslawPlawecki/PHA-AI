# PHANTOM test suite

Layout mirrors `src/phantom/`, split by test type.

## Running

```bash
pip install -e ".[dev]"

pytest                       # everything (default: single process, order shuffled)
pytest -m "not slow"         # what CI runs on every PR
pytest -m "not slow and not ml and not property"   # fast sanity (pre-commit)
pytest -n auto               # parallel (CI)
pytest --cov=phantom --cov-branch --cov-report=term-missing
pytest -p no:randomly        # disable order shuffling while debugging
pytest --randomly-seed=<n>   # reproduce a shuffled run
```

## Layout

| Dir | What | Marker (auto) |
| --- | --- | --- |
| `unit/` | pure logic, one function/class, real pandas/numpy on tiny data | — |
| `integration/` | one worker class + real filesystem (`tmp_workspace`) | `integration` |
| `controller/` | a `*Controller.run()` with prompts stubbed | `controller` |
| `regression/` | golden files: scanner logs, evaluator vectors, config snapshots | — |
| `_helpers/` | `frames.py` (DataFrame builders), `fakes.py` (`FakeEstimator`, `patch_prompts`), `assertions.py`, `workspace.py` | not collected |
| `fixtures/` | data files: `config/`, `logs/`, `golden/` | not collected |

## Markers

`slow` (real catboost/xgb/PERMANOVA, big Optuna budgets — nightly only),
`ml`, `property` (hypothesis), `integration`, `controller`,
`critical` (a business-critical path — see the testing strategy),
`flaky` (quarantined, non-blocking).

`--strict-markers` is on: register any new marker in `pyproject.toml`.

## Conventions

- **No real ML models** in `unit`/`integration` — use `FakeEstimator`. Real
  estimators only behind `@pytest.mark.slow` / `@pytest.mark.ml`.
- **No real data workspace** — the `phantom_data_root` / `tmp_workspace`
  fixtures redirect `ConfigLoader` at a `tmp_path` tree.
- Build test data with `_helpers/frames.py`, not CSV files, unless the test
  is specifically about file parsing.
- Every `raise` / early-return in a critical module gets a negative test.
- Golden files are updated by a deliberate commit, never regenerated in CI.

## Business-critical paths (`@pytest.mark.critical`)

1. Patient→label rule (`Labeling.derive_label`, `SampleThresholdLabeler`)
2. Patient-level split + test-fold column alignment (`nested_cv`, `matrix.feature_columns`)
3. `MetadataManager.add_metadata_columns` collision guard
4. CheckV mask application (`apply_mask`)
5. Feature-matrix construction (`build_matrix` binary/count + `min_patients`)
6. Merge step (`FeatureConcatenator`)
7. Metric correctness (`EvaluatorSl.evaluate`)
8. `FeatureConfigManager.create_new_version`
9. Optuna search wiring + degenerate-case returns
