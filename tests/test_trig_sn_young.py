"""Tests for trig_sn_young's mapping into the TRIGGER output format."""

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
import yaml
from astropy.time import Time

import trig_sn_young
from trig_sn_young import (
    build_targets,
    compute_merit,
    latest_detection_mags,
    parse_cli,
    tns_bandpass,
)
from trigger_output import coerce_dtypes, validate_targets


@pytest.fixture
def sample_objects() -> pd.DataFrame:
    """Three merged Alerce/TNS objects; the last is fainter than 21 mag."""
    return pd.DataFrame(
        {
            "discmjd": [61367.0, 61366.0, 61365.0],
            "ra": [10.0, 200.0, 300.0],
            "dec": [5.0, -30.0, 40.0],
            "probability": [0.9, 1.0, 0.8],
            "sel_mag_ztf_r": [18.0, np.nan, 21.5],
            "sel_mag_err_ztf_r": [0.05, np.nan, 0.2],
            "sel_mag_mjd_ztf_r": [61367.5, np.nan, 61366.0],
            "sel_mag_atlas_o": [np.nan, 17.0, np.nan],
            "sel_mag_err_atlas_o": [np.nan, np.nan, np.nan],
            "sel_mag_mjd_atlas_o": [np.nan, 61366.0, np.nan],
            "Obj. Type": [np.nan, "SN Ia", np.nan],
            "Redshift": [np.nan, 0.03, np.nan],
        },
        index=pd.Index(["ZTF26aaaaaaa", "2026abc", "ZTF26bbbbbbb"]),
    )


def test_latest_detection_mags():
    det = pd.DataFrame(
        {
            "fid": [2, 2, 1],
            "mjd": [61360.0, 61362.0, 61361.0],
            "magpsf": [19.0, 18.0, 18.5],
            "sigmapsf": [0.1, 0.05, 0.08],
        }
    )
    m = latest_detection_mags(det)
    assert m["sel_mag_ztf_r"] == 18.0
    assert m["sel_mag_mjd_ztf_r"] == 61362.0
    assert m["sel_mag_err_ztf_r"] == 0.05
    assert m["sel_mag_ztf_g"] == 18.5
    assert np.isnan(m["sel_mag_ztf_i"])


@pytest.mark.parametrize(
    "s,x",
    [
        ("r-ZTF", "ztf_r"),
        ("orange-ATLAS", "atlas_o"),
        ("cyan-ATLAS", "atlas_c"),
        ("g-ASASSN", "asassn_g"),
        ("Clear", "unknown_clear"),
        ("V-ASAS SN", "asas_sn_v"),
    ],
)
def test_tns_bandpass(s, x):
    assert tns_bandpass(s) == x


def test_merit_youngest_highest_and_floor():
    m = compute_merit(pd.Series([1.0, 1.0, 1.0]), pd.Series([0.0, 0.25, 2.0]))
    assert m.iloc[0] == m.iloc[1] == 1.0
    assert m.iloc[2] == pytest.approx(0.25)
    assert m.notna().all()


def test_build_targets_validates(sample_objects):
    t = build_targets(sample_objects, parse_cli(["-a", "7"]), now_mjd=61368.4)
    validate_targets(coerce_dtypes(t))
    assert len(t) == 2
    assert (t["observe_until"] == t["discmjd"] + 7).all()
    assert (t["observe_from"] == 61368.4).all()
    assert "Obj. Type" not in t and "Redshift" not in t
    assert {"discmjd", "probability", "age"} <= set(t.columns)
    assert list(t["sel_objid"]) == ["ZTF26aaaaaaa", "2026abc"]  # youngest first


def test_build_targets_drops_expired(sample_objects):
    # with -a 1.5 the 2-day-old object's window has already closed
    t = build_targets(sample_objects, parse_cli(["-a", "1.5"]), now_mjd=61368.4)
    assert list(t["sel_objid"]) == ["ZTF26aaaaaaa"]
    validate_targets(coerce_dtypes(t))


def test_build_targets_all_cut_gives_empty(sample_objects):
    t = build_targets(sample_objects, parse_cli(["-m", "10"]), now_mjd=61368.4)
    assert len(t) == 0
    validate_targets(coerce_dtypes(t))


def test_parse_cli_no_dec():
    with pytest.raises(SystemExit):
        parse_cli(["-d", "-20"])


def test_parse_cli_output_root_default():
    assert str(parse_cli([]).output_root) == (
        "/uufs/chpc.utah.edu/common/home/sdss10/sdss6work/due/selectors/"
        "u6079968/trig_sn_young"
    )


class _FailedResponse:
    status_code = 500
    text = ""


def test_query_tns_no_pages(monkeypatch):
    monkeypatch.setattr(
        trig_sn_young.requests, "get", lambda *a, **k: _FailedResponse()
    )
    out = trig_sn_young.query_tns(parse_cli([]))
    assert len(out) == 0
    assert {"discmjd", "ra", "dec", "probability"} <= set(out.columns)


def test_missing_output_root_fails_before_queries(monkeypatch, tmp_path):
    def boom(*args, **kwargs):
        raise AssertionError("queried before checking output root")

    monkeypatch.setattr(trig_sn_young, "query_alerce", boom)
    monkeypatch.setattr(trig_sn_young, "query_tns", boom)
    missing = tmp_path / "nope"
    with pytest.raises(OSError, match="nope"):
        trig_sn_young.main(["--output-root", str(missing)])


@pytest.mark.parametrize("n", [0, 1])
def test_deduplicate_few_objects(sample_objects, n):
    out = trig_sn_young.deduplicate(sample_objects.iloc[:n])
    assert len(out) == n


class _EmptyAlerce:
    """Mimics alerce: an empty result with index="oid" raises KeyError."""

    last_kwargs: ClassVar[dict] = {}

    def query_objects(self, **kwargs):
        _EmptyAlerce.last_kwargs = kwargs
        if "index" in kwargs:
            raise KeyError("None of ['oid'] are in the columns")
        return pd.DataFrame()


def test_query_alerce_no_objects(monkeypatch):
    monkeypatch.setattr(trig_sn_young, "Alerce", _EmptyAlerce)
    out = trig_sn_young.query_alerce(parse_cli(["-a", "7"]), now_mjd=61368.25)
    assert _EmptyAlerce.last_kwargs["firstmjd"] == [61361.25, 61368.25]
    assert len(out) == 0
    assert {"discmjd", "ra", "dec", "probability"} <= set(out.columns)


class _UntypedTNSPage:
    status_code = 200
    text = (
        "Name,RA,DEC,Obj. Type,Discovery Date (UT),Discovery Filter,Discovery Mag/Flux\n"
        "AT 2026xyz,01:00:00.000,+10:00:00.00,,2026-11-22 03:00:00.000,orange-ATLAS,18.2\n"
    )


def test_query_tns_untyped_page(monkeypatch):
    responses = iter([_UntypedTNSPage(), _FailedResponse()])
    monkeypatch.setattr(trig_sn_young.requests, "get", lambda *a, **k: next(responses))
    monkeypatch.setattr(trig_sn_young.time, "sleep", lambda s: None)
    out = trig_sn_young.query_tns(parse_cli([]))
    assert len(out) == 0


def test_main_empty_night_writes_header_only(monkeypatch, tmp_path):
    empty = trig_sn_young.empty_shared_frame
    monkeypatch.setattr(trig_sn_young, "query_alerce", lambda args, now_mjd: empty())
    monkeypatch.setattr(trig_sn_young, "query_tns", lambda args: empty())
    trig_sn_young.main(["--output-root", str(tmp_path)])
    (run_dir,) = tmp_path.iterdir()
    csv = next(run_dir.glob("*_targets.csv"))
    assert len(pd.read_csv(csv)) == 0
    assert "merit" in pd.read_csv(csv).columns
    assert next(run_dir.glob("*_meta.yaml")).exists()


def test_main_uses_one_start_time(monkeypatch, tmp_path):
    start = Time(61368.25, format="mjd")  # 2026-11-24 06:00 UTC
    monkeypatch.setattr(trig_sn_young.Time, "now", staticmethod(lambda: start))
    seen = {}

    def fake_alerce(args, now_mjd):
        seen["alerce"] = now_mjd
        return trig_sn_young.empty_shared_frame()

    def fake_build(objects, args, now_mjd):
        seen["build"] = now_mjd
        return build_targets(objects, args, now_mjd)

    monkeypatch.setattr(trig_sn_young, "query_alerce", fake_alerce)
    monkeypatch.setattr(
        trig_sn_young, "query_tns", lambda args: trig_sn_young.empty_shared_frame()
    )
    monkeypatch.setattr(trig_sn_young, "build_targets", fake_build)
    trig_sn_young.main(["--output-root", str(tmp_path)])

    assert seen == {"alerce": start.mjd, "build": start.mjd}
    (run_dir,) = tmp_path.iterdir()
    assert run_dir.name == "61368"
    meta = yaml.safe_load(next(run_dir.glob("*_meta.yaml")).read_text())
    assert meta["selector_run"]["time_started"] == "2026-11-23T23:00:00-07:00"
