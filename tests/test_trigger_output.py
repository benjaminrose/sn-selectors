"""Tests for the SDSS-VI TRIGGER output-format module."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import trigger_output
from trigger_output import (
    add_spec_defaults,
    coerce_dtypes,
    git_tag,
    mag_columns,
    opt_brightness_class,
    run_paths,
    utah_isotime,
    validate_targets,
    write_run,
)


def test_opt_brightness_thresholds():
    mags = (12.9, 13.0, 16.0, 16.1, np.nan)
    assert [opt_brightness_class(m) for m in mags] == ["B", "I", "I", "F", "F"]


def test_mag_columns_excludes_err_and_mjd():
    df = pd.DataFrame(
        columns=["sel_mag_ztf_r", "sel_mag_err_ztf_r", "sel_mag_mjd_ztf_r", "ra"]
    )
    assert mag_columns(df) == ["sel_mag_ztf_r"]


def test_add_spec_defaults_uses_brightest_mag():
    df = pd.DataFrame(
        {"sel_mag_ztf_g": [17.0, np.nan], "sel_mag_ztf_r": [15.0, np.nan]}
    )
    out = add_spec_defaults(df)
    assert list(out["opt_brightness"]) == ["I", "F"]
    assert (out["n_exposures"] == 4).all()
    assert out["dark_only"].all()
    assert (out["fiber_type"] == "BOSS").all()
    assert (out["ir_brightness"] == "none").all()


def test_validate_accepts_good_frame(good_targets):
    validate_targets(coerce_dtypes(good_targets))


@pytest.mark.parametrize(
    "col,value",
    [
        ("merit", 1.2),
        ("merit", -0.1),
        ("merit", np.nan),
        ("sel_objid", "x" * 33),
        ("opt_brightness", "Q"),
        ("fiber_type", "LAMOST"),
    ],
)
def test_validate_rejects_bad_values(good_targets, col, value):
    good_targets.loc[0, col] = value
    with pytest.raises(ValueError, match=col):
        validate_targets(good_targets)


def test_validate_rejects_missing_column(good_targets):
    with pytest.raises(ValueError, match="merit"):
        validate_targets(good_targets.drop(columns="merit"))


def test_validate_rejects_duplicate_objid(good_targets):
    good_targets.loc[1, "sel_objid"] = good_targets.loc[0, "sel_objid"]
    with pytest.raises(ValueError, match="sel_objid"):
        validate_targets(good_targets)


def test_validate_rejects_window_inverted(good_targets):
    good_targets.loc[0, "observe_until"] = good_targets.loc[0, "observe_from"] - 1
    with pytest.raises(ValueError, match="observe_until"):
        validate_targets(good_targets)


def test_validate_accepts_empty_frame(good_targets):
    validate_targets(coerce_dtypes(good_targets.iloc[:0]))


def test_coerce_dtypes(good_targets):
    out = coerce_dtypes(good_targets)
    assert out["ra"].dtype == np.float64
    assert out["merit"].dtype == np.float32
    assert out["n_exposures"].dtype == np.int32
    assert out["dark_only"].dtype == np.bool_
    assert out["sel_mag_err_ztf_r"].dtype == np.float32


# --- Task 2: paths, git tag, metadata and writing ---------------------------

WRITE_KW = {
    "selector": "trig_sn_young",
    "repo": "https://github.com/benjaminrose/sn-selectors",
    "chpc_username": "u6079968",
    "tag": "v0.0.0",
    "time_started": "2026-11-23T12:00:00-07:00",
    "cmd": "trig_sn_young.py -a 7",
}


def test_run_paths():
    p = run_paths(Path("/r"), "trig_sn_young", 61368)
    assert p.run_dir == Path("/r/61368")
    assert p.targets == Path("/r/61368/trig_sn_young_61368_targets.csv")
    assert p.meta == Path("/r/61368/trig_sn_young_61368_meta.yaml")
    assert p.log == Path("/r/61368/61368.log")


def test_git_tag_falls_back(tmp_path):
    assert git_tag(tmp_path) == "unknown"


def test_git_tag_in_repo():
    assert git_tag(Path(trigger_output.__file__).parent) not in ("", "unknown")


def test_utah_isotime_offset():
    t = datetime(2026, 1, 15, 19, 0, tzinfo=UTC)
    assert utah_isotime(t) == "2026-01-15T12:00:00-07:00"


def test_write_run_roundtrip(tmp_path, good_targets):
    paths = run_paths(tmp_path, "trig_sn_young", 61368)
    write_run(good_targets, paths, **WRITE_KW)
    csv = pd.read_csv(paths.targets)
    assert set(trigger_output.REQUIRED_DTYPES) <= set(csv.columns)
    assert len(csv) == 2
    meta = yaml.safe_load(paths.meta.read_text())
    run = meta["selector_run"]
    assert run["datafile"] == paths.targets.name
    assert run["selector"] == "trig_sn_young"
    assert run["chpc_username"] == "u6079968"
    assert run["tag"] == "v0.0.0"
    assert "time_datafile" in run and "time_started" in run and "cmd" in run
    assert meta["bandpasses"]["ztf_r"]["lambda_eff"] > 0


def test_write_run_empty_targets(tmp_path, good_targets):
    paths = run_paths(tmp_path, "trig_sn_young", 61368)
    write_run(good_targets.iloc[:0], paths, **WRITE_KW)
    csv = pd.read_csv(paths.targets)
    assert len(csv) == 0
    assert "merit" in csv.columns
    assert paths.meta.exists()


def test_write_run_invalid_writes_nothing(tmp_path, good_targets):
    good_targets.loc[0, "merit"] = 2.0
    paths = run_paths(tmp_path, "trig_sn_young", 61368)
    with pytest.raises(ValueError):
        write_run(good_targets, paths, **WRITE_KW)
    assert not paths.targets.exists()
    assert not paths.meta.exists()


def test_write_run_invalid_logs_error(tmp_path, good_targets, caplog):
    good_targets.loc[0, "merit"] = 2.0
    paths = run_paths(tmp_path, "trig_sn_young", 61368)
    with pytest.raises(ValueError):
        write_run(good_targets, paths, **WRITE_KW)
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert errors and "merit" in errors[0].getMessage()
