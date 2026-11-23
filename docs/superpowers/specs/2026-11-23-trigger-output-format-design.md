# Design: SDSS-VI TRIGGER output format for `trig_sn_young`

Date: 2026-11-23. Status: approved 2026-11-23.

## Intent

The SDSS-VI TRIGGER spec ("TRIGGER Selectors output format", Tom Dwelly, #due-trigger) changed the target file
columns, added a per-run metadata yaml, and moved selector outputs to a Collector-managed SAS directory.
`trig_sn_young.py` must produce output the Collector can ingest. The spec is still being finalised, so the
spec-specific logic should live in one place that is easy to update.

## Goals

- `trig_sn_young` writes one spec-compliant targets CSV and one metadata yaml per run.
- Spec logic (columns, brightness classes, magnitude columns, validation, paths, metadata) is reusable by the
  planned `trig_sn_ia` and `trig_sn_host_redshifts` selectors.
- Offline tests cover the spec logic.

## Non-goals

- Implementing `trig_sn_ia` or `trig_sn_host_redshifts`.
- Changing the science selection (Alerce/TNS queries, dedup, mag cut, merit formula), except the Alerce
  detection bug below.
- FITS/Parquet/HDF5 output.

## Anti-goals

- Writing a file the Collector would reject or misread. Validation failures raise; nothing is written.
- Hardcoding secrets. (The CHPC username in the default path is accepted by the owner.)

## Approaches

- **A. Shared `trigger_output.py` module (chosen).** Selector-independent spec logic in one module next to the
  scripts, imported like `fastdb_client.py`. Spec changes land in one file; future selectors reuse it.
- B. Everything inside `trig_sn_young.py`. Smaller diff, but each new selector copies the spec logic.

## Decisions (from the owner)

| Topic | Decision |
|---|---|
| Selector mode | Singular: each run is the full list. |
| Telescope split | Removed. One targets file per run; `--dec` option removed. |
| Run timestamp | Integer MJD. One run per night; a rerun overwrites. |
| Output location | `--output-root` CLI option = everything up to the MJD directory. Default `/uufs/chpc.utah.edu/common/home/sdss10/sdss6work/due/selectors/u6079968/trig_sn_young`. `ROOT_LOC` is replaced by it. |
| Files per run | `<root>/<MJD>/trig_sn_young_<MJD>_targets.csv`, `trig_sn_young_<MJD>_meta.yaml`, `<MJD>.log`. |
| `tag` | `git describe --tags --always --dirty` run in the script's directory (falls back to commit hash); `"unknown"` if git fails. |
| `repo` | `https://github.com/benjaminrose/sn-selectors` |
| `chpc_username` | `u6079968` |
| `dark_only` | `True` for all targets. **[OPEN]** see below. |
| `ir_brightness` | Literal string `"none"`. **[OPEN]** see below. |
| `observe_from` / `observe_until` | Run MJD / `discmjd + age` (`-a`, default 7 d): observed until `age` days post discovery. |
| `merit` | `probability / max(age, 0.5 d)`, normalized to max 1. Ordering checked: youngest gets the highest merit. |
| Alerce mags | Use the **most recent** detection per band (fixes current code, which takes the first). `sel_mag_mjd_{x}` = that detection's MJD. |

## Builder's calls (stated as assumptions; owner may override)

- `fiber_type = "BOSS"`, `n_exposures = 4` for all targets.
- `sel_objid` = ZTF oid or TNS name (already the DataFrame index); validated ≤ 32 characters.
- `opt_brightness` from the brightest recorded `sel_mag_*`, spec thresholds: B < 13 ≤ I ≤ 16 < F. No mag → `"F"`.
- Bandpass name `x` = `{survey}_{band}`: Alerce → `ztf_g`, `ztf_r`, `ztf_i`. TNS `"<band>-<system>"` →
  `{system}_{band}` lowercased, with ATLAS `orange`/`cyan` → `atlas_o`/`atlas_c`.
- `sel_mag_err_{x}`: Alerce `sigmapsf`; TNS has none → NaN. TNS `sel_mag_mjd_{x}` = `discmjd`.
- The `-m` mag cut keeps its current rule (any band brighter than the limit), now applied to latest mags.
- Meta yaml: `time_started` / `time_datafile` are ISO 8601 in America/Denver; `datafile` is the bare filename
  (relative to the yaml); `cmd` is the full `sys.argv` joined; `bandpasses:` stanza lists only bands present,
  with `system: AB`, `mag_type: psf`, `err_metric: 1sigma` (Alerce) and `lambda_eff` in nm taken from the SVO
  Filter Profile Service (values cited in the code, not invented).
- CSV dtypes are enforced before writing (`float64` ra/dec, `float32` others, `int32` n_exposures, `bool`
  dark_only); output stays CSV.
- Logging is configured after CLI parsing, since the log path depends on `--output-root`.
- New dependency `pyyaml` in the PEP 723 header; `pytest`-based tests in `tests/` (run with
  `uv run --with pytest ...`), plus `ruff` and `mypy`.
- `sbatch_trig_sn.sh` needs no change beyond any flags; `README.md` and `CLAUDE.md` are updated.

## Module boundary

`trigger_output.py` (no network, pure functions where possible):
- brightness class from mags; build `sel_mag_*` columns; required-column defaults; dtype coercion;
  `validate_targets()` (required columns present, `merit` in [0,1], `sel_objid` ≤ 32 chars and unique, allowed
  string values, `observe_from ≤ observe_until`); run-directory and file paths; `write_run(targets, meta, ...)`
  that validates then writes CSV and yaml; `git_tag()`.

`trig_sn_young.py`: CLI, queries, dedup, mag cut, merit, mapping its columns into the spec frame, calling
`write_run`.

## Open items

1. **[OPEN] `dark_only`:** can sufficiently bright transients be observed in bright time? Currently `True` for all.
2. **[OPEN] `ir_brightness`:** spec allows only B/I/F; we write `"none"`. Confirm with Tom whether the Collector
   accepts it, or get IR thresholds.

## Resolved at approval (2026-11-23)

- **Extra columns:** keep `discmjd`, `probability`, `age` in the targets file for traceability; drop TNS
  `Obj. Type` and `Redshift`. The spec does not say whether extras are allowed; revisit if the Collector objects.
- **Merit near age 0:** `merit = probability / max(age, 0.5 d)`, normalized to max 1. Keeps the youngest-first
  ordering, caps how much one object can dominate, and removes the divide-by-zero.
