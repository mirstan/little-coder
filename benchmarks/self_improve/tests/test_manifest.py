import dataclasses

import pytest
import yaml

from benchmarks.self_improve.manifest import MANIFEST_SCHEMA_VERSION, Manifest


def _manifest(**overrides) -> Manifest:
    fields = dict(
        created_at="2026-09-26T12:00:00+00:00",
        seed=42,
        benchmark="aider_polyglot",
        language="python",
        splits={"search": ["wordy", "acronym"], "acceptance": ["leap"], "test": ["bob"]},
        searchable_components=["agents_md"],
        sampling={"temperature": 0.3},
        budget={"max_metric_calls": 20, "max_wall_clock_s": 14400.0},
        env_fingerprint={"model": "gpt-fake"},
    )
    fields.update(overrides)
    return Manifest(**fields)


def test_defaults_are_the_pre_registered_protocol_constants():
    m = _manifest()
    assert m.schema_version == MANIFEST_SCHEMA_VERSION
    assert m.metrics["primary"] == "pass"
    assert m.metrics["efficiency"] == "decode+prompt tokens"
    assert (m.alpha_t4, m.root_tolerance_pp, m.mde, m.t5_max_candidates) == (0.01, 2.0, 0.2, 3)


def test_save_load_round_trip(tmp_path):
    m = _manifest()
    path = tmp_path / "manifest.yaml"
    m.save(path)
    loaded = Manifest.load(path)
    assert loaded == m
    # created_at must survive as a string -- an unquoted ISO timestamp would
    # come back from safe_load as a datetime and break equality.
    assert isinstance(loaded.created_at, str)


@pytest.mark.parametrize("splits,match", [
    ({"search": ["a"], "acceptance": ["a"], "test": ["b"]}, "disjoint"),
    ({"search": ["a"], "acceptance": ["b"], "test": ["a"]}, "disjoint"),
    ({"search": ["a"], "acceptance": [], "test": ["b"]}, "empty"),
    ({"search": ["a"], "acceptance": ["b"]}, "search, acceptance, test"),
])
def test_splits_must_be_three_non_empty_disjoint_lists(splits, match):
    with pytest.raises(ValueError, match=match):
        _manifest(splits=splits)


@pytest.mark.parametrize("sampling", [{"temperature": 0}, {"temperature": -0.1}, {}])
def test_temperature_must_be_given_and_positive(sampling):
    with pytest.raises(ValueError, match="temperature"):
        _manifest(sampling=sampling)


def test_t5_max_candidates_must_be_at_least_one():
    with pytest.raises(ValueError, match="t5_max_candidates"):
        _manifest(t5_max_candidates=0)


@pytest.mark.parametrize("field", ["alpha_t4", "mde"])
@pytest.mark.parametrize("value", [0.0, 1.0, -0.5])
def test_alphas_must_be_in_the_open_unit_interval(field, value):
    with pytest.raises(ValueError, match=field):
        _manifest(**{field: value})


def test_load_rejects_unknown_fields_and_other_schema_versions(tmp_path):
    data = dataclasses.asdict(_manifest())
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump({**data, "surprise": 1}))
    with pytest.raises(ValueError, match="surprise"):
        Manifest.load(path)
    path.write_text(yaml.safe_dump({**data, "schema_version": MANIFEST_SCHEMA_VERSION + 1}))
    with pytest.raises(ValueError, match="schema_version"):
        Manifest.load(path)


def test_save_refuses_to_overwrite_an_existing_manifest(tmp_path):
    """The manifest pre-registers a run, so it is write-once: a second save
    to the same path must fail loudly and leave the original bytes."""
    path = tmp_path / "manifest.yaml"
    _manifest(seed=1).save(path)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        _manifest(seed=2).save(path)
    assert path.read_bytes() == before
