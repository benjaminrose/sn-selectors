# TRIGGER Output Format Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `trig_sn_young.py` write one SDSS-VI TRIGGER spec-compliant targets CSV plus a metadata yaml per run, with the spec logic in a reusable `trigger_output.py`.

**Architecture:** `trigger_output.py` holds selector-independent spec logic (brightness classes, defaults, dtypes, validation, paths, git tag, bandpass stanza, writing). `trig_sn_young.py` keeps the science (queries, dedup, mag cut, merit) and maps its frame into spec columns before calling `write_run`.

**Tech Stack:** Python ≥ 3.12, pandas, numpy, astropy, alerce, pyyaml, pytest; PEP 723 scripts run with `uv`.

**Spec:** `docs/superpowers/specs/2026-11-23-trigger-output-format-design.md`

## Global Constraints

- Selector name `trig_sn_young`; files `trig_sn_young_<MJD>_targets.csv`, `trig_sn_young_<MJD>_meta.yaml`, `<MJD>.log` in `<output-root>/<MJD>/` (integer MJD).
- `--output-root` default: `/uufs/chpc.utah.edu/common/home/sdss10/sdss6work/due/selectors/u6079968/trig_sn_young`.
- `repo: https://github.com/benjaminrose/sn-selectors`; `chpc_username: u6079968`.
- Required target columns and dtypes: `sel_objid` str (≤ 32 chars), `ra`/`dec` float64, `merit` float32 in [0, 1], `n_exposures` int32 (= 4), `dark_only` bool (= True), `opt_brightness` str in {B, I, F}, `ir_brightness` str (= `"none"`), `fiber_type` str (= `"BOSS"`), `observe_from`/`observe_until` float32.
- `opt_brightness` from brightest `sel_mag_{x}`: B if mag < 13, I if 13 ≤ mag ≤ 16, F if mag > 16 or no mag.
- Magnitude triples: `sel_mag_{x}`, `sel_mag_err_{x}`, `sel_mag_mjd_{x}` (float32); `x` = `{survey}_{band}` lowercased.
- Extra columns kept: `discmjd`, `probability`, `age`. Dropped: TNS `Obj. Type`, `Redshift`, and old `program`, `sky_brightness_mode`, `active`, `priority`.
- `merit = probability / max(age, 0.5)`, normalized to max 1.
- `observe_from` = decimal MJD at run start; `observe_until = discmjd + args.age`.
- Times in meta: ISO 8601, America/Denver. `datafile`: bare filename.
- No `print`; logging to the run log only. `fastdb_client.py` untouched. Commit only when the owner asks; repo allows commits on `main`, one per task.
- Test command (used below as `$PYTEST`): `uv run --with pytest --with pandas --with numpy --with pyyaml --with astropy --with alerce --with requests pytest`

## Review Focus

1. **Zero targets after cuts:** a singular selector must still write a header-only CSV and the meta yaml, so the Collector deactivates the previous list. Test in Task 2 and Task 3.
2. **TNS returns no pages** (non-200 on page 0): today `pd.concat([])` raises. Expect an empty frame with the shared schema and a warning. Test in Task 3.
3. **Output root missing or unwritable** (directory not yet created by Tom): fail before any network query with a clear `OSError` naming the path. Test in Task 3.
4. **TNS filter string without `-`** (e.g. `"Clear"`) or with spaces: bandpass name must still be a safe lowercase identifier (`[a-z0-9_]`). Test in Task 3.
5. **Duplicate `sel_objid`** after concat: validation must reject it rather than write a file. Test in Task 1.

---

### Task 1: Spec columns, dtypes and validation in `trigger_output.py`

**Files:**
- Create: `trigger_output.py`
- Test: `tests/test_trigger_output.py`

**Interfaces:**
- Produces:
  - `REQUIRED_DTYPES: dict[str, str]` (column → numpy dtype name per Global Constraints; strings as `"object"`)
  - `mag_columns(df: pd.DataFrame) -> list[str]` — `sel_mag_{x}` only, not `_err_`/`_mjd_`
  - `opt_brightness_class(mag: float) -> str`
  - `add_spec_defaults(targets: pd.DataFrame, *, n_exposures: int = 4, dark_only: bool = True, fiber_type: str = "BOSS", ir_brightness: str = "none") -> pd.DataFrame` — adds those columns and `opt_brightness` from the brightest mag
  - `coerce_dtypes(targets: pd.DataFrame) -> pd.DataFrame` — required columns per `REQUIRED_DTYPES`, all `sel_mag*` to float32
  - `validate_targets(targets: pd.DataFrame) -> None` — raises `ValueError` listing every problem found

- [ ] **Step 1: Write failing tests**

```python
def test_opt_brightness_thresholds():
    assert [opt_brightness_class(m) for m in (12.9, 13.0, 16.0, 16.1, np.nan)] == ["B", "I", "I", "F", "F"]

def test_mag_columns_excludes_err_and_mjd():
    df = pd.DataFrame(columns=["sel_mag_ztf_r", "sel_mag_err_ztf_r", "sel_mag_mjd_ztf_r", "ra"])
    assert mag_columns(df) == ["sel_mag_ztf_r"]

def test_add_spec_defaults_uses_brightest_mag():
    df = pd.DataFrame({"sel_mag_ztf_g": [17.0, np.nan], "sel_mag_ztf_r": [15.0, np.nan]})
    out = add_spec_defaults(df)
    assert list(out["opt_brightness"]) == ["I", "F"]
    assert (out["n_exposures"] == 4).all() and out["dark_only"].all()
    assert (out["fiber_type"] == "BOSS").all() and (out["ir_brightness"] == "none").all()

def test_validate_accepts_good_frame(good_targets): validate_targets(good_targets)

@pytest.mark.parametrize("col,value", [("merit", 1.2), ("merit", -0.1), ("sel_objid", "x" * 33),
                                       ("opt_brightness", "Q"), ("fiber_type", "LAMOST")])
def test_validate_rejects_bad_values(good_targets, col, value): ...  # set row 0, expect ValueError naming col

def test_validate_rejects_missing_column(good_targets): ...  # drop "merit" -> ValueError
def test_validate_rejects_duplicate_objid(good_targets): ...  # two rows same sel_objid -> ValueError
def test_validate_rejects_window_inverted(good_targets): ...  # observe_from > observe_until -> ValueError
def test_validate_accepts_empty_frame(good_targets): validate_targets(coerce_dtypes(good_targets.iloc[:0]))
def test_coerce_dtypes(good_targets): ...  # ra float64, merit float32, n_exposures int32, dark_only bool
```

`good_targets` fixture: two rows with every required column valid (`ir_brightness="none"` is allowed; `ir_brightness` is not checked against B/I/F per spec decision).

- [ ] **Step 2: Run** `$PYTEST tests/test_trigger_output.py -v` — expect import errors / FAIL.
- [ ] **Step 3: Implement** the Produces list in `trigger_output.py` with numpy-style docstrings and type hints. `validate_targets` collects all problems then raises once.
- [ ] **Step 4: Run** `$PYTEST tests/test_trigger_output.py -v` — all PASS; `ruff check trigger_output.py tests` and `mypy trigger_output.py` clean.
- [ ] **Step 5: Commit** (when owner asks): `git add trigger_output.py tests/test_trigger_output.py` → `Add TRIGGER spec columns and validation module`

### Task 2: Paths, git tag, bandpasses and `write_run`

**Files:**
- Modify: `trigger_output.py`
- Test: `tests/test_trigger_output.py`

**Interfaces:**
- Consumes: Task 1 functions.
- Produces:
  - `@dataclass(frozen=True) class RunPaths: run_dir: Path; targets: Path; meta: Path; log: Path`
  - `run_paths(output_root: Path, selector: str, mjd: int) -> RunPaths` — pure, no filesystem access
  - `git_tag(repo_dir: Path) -> str` — `git describe --tags --always --dirty` with `cwd=repo_dir`; `"unknown"` on any failure
  - `utah_isotime(t: datetime | None = None) -> str` — ISO 8601 in `ZoneInfo("America/Denver")`, seconds precision with offset
  - `KNOWN_BANDPASSES: dict[str, dict[str, object]]` — `ztf_g`, `ztf_r`, `ztf_i`, `atlas_o`, `atlas_c` with `system`, `mag_type: psf`, `lambda_eff` (nm, from the SVO Filter Profile Service; cite the SVO filter IDs in a comment; look the values up, do not estimate)
  - `bandpass_stanza(targets: pd.DataFrame) -> dict[str, dict[str, object]]` — only bands present in `mag_columns`; unknown band → `{}` plus a logged warning
  - `write_run(targets: pd.DataFrame, paths: RunPaths, *, selector: str, repo: str, chpc_username: str, tag: str, time_started: str, cmd: str) -> None` — `coerce_dtypes`, `validate_targets`, write CSV (`index=False`), then yaml with the `selector_run:` keys from the spec (`selector`, `repo`, `tag`, `time_started`, `time_datafile`, `chpc_username`, `datafile`, `cmd`) and top-level `bandpasses:`

- [ ] **Step 1: Write failing tests**

```python
def test_run_paths():
    p = run_paths(Path("/r"), "trig_sn_young", 61368)
    assert p.targets == Path("/r/61368/trig_sn_young_61368_targets.csv")
    assert p.meta == Path("/r/61368/trig_sn_young_61368_meta.yaml")
    assert p.log == Path("/r/61368/61368.log")

def test_git_tag_falls_back(tmp_path): assert git_tag(tmp_path) == "unknown"   # not a repo
def test_git_tag_in_repo(): assert git_tag(Path(trigger_output.__file__).parent) not in ("", "unknown")

def test_utah_isotime_offset():
    assert utah_isotime(datetime(2026, 1, 15, 19, 0, tzinfo=UTC)) == "2026-01-15T12:00:00-07:00"

def test_write_run_roundtrip(tmp_path, good_targets):
    # write_run then read back: CSV columns ⊇ REQUIRED_DTYPES; yaml selector_run.datafile == targets filename;
    # yaml has time_datafile; bandpasses has "ztf_r" with lambda_eff

def test_write_run_empty_targets(tmp_path, good_targets):
    # zero rows -> CSV with header only, yaml written   (Review Focus 1)

def test_write_run_invalid_writes_nothing(tmp_path, good_targets):
    # merit = 2 -> ValueError and neither file exists
```

- [ ] **Step 2: Run** `$PYTEST tests/test_trigger_output.py -v` — new tests FAIL.
- [ ] **Step 3: Implement** the Produces list. `write_run` creates `paths.run_dir` with `mkdir(parents=False, exist_ok=True)` only if missing (the root must already exist).
- [ ] **Step 4: Run** tests, `ruff check`, `mypy trigger_output.py` — all clean.
- [ ] **Step 5: Commit** (when owner asks) → `Add run paths, metadata yaml and writer to TRIGGER module`

### Task 3: Port `trig_sn_young.py` to the new format

**Files:**
- Modify: `trig_sn_young.py` (whole file), `README.md`, `CLAUDE.md`
- Test: `tests/test_trig_sn_young.py`

**Interfaces:**
- Consumes: everything from Tasks 1–2.
- Produces (in `trig_sn_young.py`, importable without side effects):
  - `parse_cli(argv: list[str] | None = None) -> argparse.Namespace` — keeps `-a/-p/-m`, removes `-d/--dec`, adds `--output-root` (Path, default in Global Constraints)
  - `setup_logging(log_file: Path) -> None` — file-only DEBUG handler, called after `parse_cli`
  - `latest_detection_mags(detections: pd.DataFrame) -> dict[str, float]` — keys `sel_mag_ztf_{g,r,i}`, `sel_mag_err_ztf_*` (`sigmapsf`), `sel_mag_mjd_ztf_*` (`mjd`) from the **latest** detection per `fid` (1 g, 2 r, 3 i); missing band → NaN
  - `tns_bandpass(filter_str: str) -> str` — `"r-ZTF"`→`"ztf_r"`, `"orange-ATLAS"`→`"atlas_o"`, `"cyan-ATLAS"`→`"atlas_c"`, `"g-ASASSN"`→`"asassn_g"`, no hyphen → `"unknown_<band>"`; result matches `^[a-z0-9_]+$`
  - `compute_merit(probability: pd.Series, age: pd.Series, age_floor: float = 0.5) -> pd.Series`
  - `build_targets(objects: pd.DataFrame, args: argparse.Namespace, now_mjd: float) -> pd.DataFrame` — mag cut on `mag_columns`, `age`, `merit`, `observe_from = now_mjd`, `observe_until = discmjd + args.age`, `sel_objid` from index, `add_spec_defaults`, keep `discmjd`, `probability`, `age`, drop TNS `Obj. Type`/`Redshift`, sort by merit descending
- Shared schema for `query_alerce`/`query_tns` changes from `<filter>_mag` to the `sel_mag*` triples; both still return `discmjd`, `ra`, `dec`, `probability` indexed by object name. `query_tns` sets `sel_mag_mjd_{x} = discmjd` and `sel_mag_err_{x} = NaN`.

- [ ] **Step 1: Write failing tests**

```python
def test_latest_detection_mags():
    det = pd.DataFrame({"fid": [2, 2, 1], "mjd": [61360.0, 61362.0, 61361.0],
                        "magpsf": [19.0, 18.0, 18.5], "sigmapsf": [0.1, 0.05, 0.08]})
    m = latest_detection_mags(det)
    assert m["sel_mag_ztf_r"] == 18.0 and m["sel_mag_mjd_ztf_r"] == 61362.0 and m["sel_mag_err_ztf_r"] == 0.05
    assert m["sel_mag_ztf_g"] == 18.5 and np.isnan(m["sel_mag_ztf_i"])

@pytest.mark.parametrize("s,x", [("r-ZTF", "ztf_r"), ("orange-ATLAS", "atlas_o"), ("cyan-ATLAS", "atlas_c"),
                                 ("g-ASASSN", "asassn_g"), ("Clear", "unknown_clear")])
def test_tns_bandpass(s, x): assert tns_bandpass(s) == x          # Review Focus 4

def test_merit_youngest_highest_and_floor():
    m = compute_merit(pd.Series([1.0, 1.0, 1.0]), pd.Series([0.0, 0.25, 2.0]))
    assert m.iloc[0] == m.iloc[1] == 1.0 and m.iloc[2] == pytest.approx(0.25) and m.notna().all()

def test_build_targets_validates(sample_objects):    # 3 objects incl. one too faint
    t = build_targets(sample_objects, parse_cli(["-a", "7"]), now_mjd=61368.4)
    validate_targets(coerce_dtypes(t)); assert len(t) == 2
    assert (t["observe_until"] == t["discmjd"] + 7).all() and (t["observe_from"] == 61368.4).all()
    assert "Obj. Type" not in t and {"discmjd", "probability", "age"} <= set(t.columns)

def test_build_targets_all_cut_gives_empty(sample_objects): ...   # -m 10 -> zero rows, still validates (RF 1)
def test_parse_cli_no_dec(): with pytest.raises(SystemExit): parse_cli(["-d", "-20"])
def test_query_tns_no_pages(monkeypatch): ...  # requests.get -> status 500; expect empty frame, no raise (RF 2)
def test_missing_output_root_fails_before_queries(monkeypatch, tmp_path): ...
    # main([... "--output-root", str(tmp_path / "nope")]) raises OSError naming the path; query_alerce not called (RF 3)
```

- [ ] **Step 2: Run** `$PYTEST tests/test_trig_sn_young.py -v` — FAIL.
- [ ] **Step 3: Implement.** Remove import-time `NOW`/logging side effects: a `main(argv: list[str] | None = None) -> None` records `time_started = utah_isotime()` and `now = Time.now()`, parses CLI, checks `args.output_root.is_dir()` (else `OSError`), builds `run_paths`, `setup_logging`, queries, concat, `deduplicate`, `build_targets`, then `write_run(..., tag=git_tag(Path(__file__).parent), cmd=" ".join(sys.argv), selector="trig_sn_young", repo=..., chpc_username="u6079968")`. Add `pyyaml` and `pandas`, `numpy`, `requests` to the PEP 723 header. Delete `clean_save`.
- [ ] **Step 4: Run** full `$PYTEST tests -v`, `ruff check .`, `mypy trigger_output.py trig_sn_young.py` (exclude `fastdb_client.py`). Then a live smoke run: `./trig_sn_young.py --output-root scratch/out` after `mkdir -p scratch/out`; inspect the CSV header and yaml.
- [ ] **Step 5: Docs.** `README.md`: output location and format. `CLAUDE.md`: replace the per-night `ROOT_LOC` convention, the shared schema (`sel_mag*` triples), the APO/LCO split, and add the test command and `scratch/` to `.gitignore` if missing.
- [ ] **Step 6: Commit** (when owner asks) → `Port trig_sn_young to SDSS-VI TRIGGER output format`

## After the plan

- Owner asks Tom Dwelly whether `ir_brightness="none"` and extra columns are accepted, and confirms `selectors/u6079968/` exists on CHPC.
- Tag `v0.1.0` once merged so `git describe` reports a version.
