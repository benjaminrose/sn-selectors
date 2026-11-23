# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

SDSS-VI transient "TRIGGER" selectors for supernova follow-up. Each `trig_sn_*.py` script is a standalone selector that
produces target lists for the TRIGGER Collector. The output format follows the SDSS wiki page "TRIGGER Selectors output
format"; the design for this repo is `docs/superpowers/specs/2026-11-23-trigger-output-format-design.md`, which also
lists open questions to revisit as the spec is finalised.

- `trig_sn_young.py`: working selector for young SNe (Alerce + TNS). Runs nightly on Slurm. Singular selector mode:
  each run is the complete target list.
- `trigger_output.py`: selector-independent spec logic (required columns and dtypes, brightness classes, validation,
  run paths, `git describe` tag, metadata yaml, `write_run`). Spec changes go here; new selectors should use it.
- `trig_sn_ia.py`: exploratory FastDB query, not a selector yet.
- `trig_sn_host_redshifts.py`: empty placeholder for a host-galaxy redshift fiber filler.
- `fastdb_client.py`: vendored third-party FASTDB client (LBNL BSD license, from rknop/rkwebutil). Treat as external;
  do not refactor it. It reads credentials from `~/.fastdb.ini`.
- `alerce_classifiers.md` (untracked): reference dump of Alerce classifier names and classes.

## Running and testing

Each selector is a PEP 723 inline-metadata script run with `uv` (shebang `#!/usr/bin/env -S uv run --script`);
dependencies live in the `# /// script` header, so add new imports there. `trigger_output.py` is imported from the
script's directory, so its imports must also be in each selector's header.

```bash
./trig_sn_young.py --help
mkdir -p scratch/out && ./trig_sn_young.py --output-root scratch/out   # local run
uv run --with pytest --with pandas --with numpy --with pyyaml --with astropy --with alerce --with requests --with scipy pytest tests
ruff check . && uv run --with mypy --with pandas-stubs --with types-PyYAML --with types-requests mypy --ignore-missing-imports trigger_output.py trig_sn_young.py
```

Tests are offline (no Alerce/TNS calls). Production runs on Utah CHPC via `sbatch sbatch_trig_sn.sh`, which does
`module load uv/sdss` and calls the script by its absolute path on the cluster.

## trig_sn_young pipeline

`query_alerce` (ZTF objects from the `stamp_classifier_2025_beta` SN class, most recent detection per g/r/i) and
`query_tns` (TNS CSV search for ZTF/ATLAS/ASAS-SN reports typed SN, `probability` fixed to 1.0, discovery mag only)
each return a DataFrame indexed by object name with a shared schema: `discmjd`, `ra`, `dec`, `probability`, and
`sel_mag_{x}` / `sel_mag_err_{x}` / `sel_mag_mjd_{x}` triples where `x` is `{survey}_{band}` (e.g. `ztf_r`,
`atlas_o`). They are concatenated, `deduplicate` drops sky matches within 1 arcsec, and `build_targets` applies the mag
cut, sets `observe_from` = run MJD and `observe_until` = `discmjd + age`, drops expired objects, computes
`merit = probability / max(age, 0.5 d)` (normalized to max 1), and adds the fixed columns. `write_run` validates and
writes.

Conventions that matter when editing:

- New data sources must return the shared schema above.
- Outputs go to `--output-root` (default the CHPC `sdss6work/due/selectors/u6079968/trig_sn_young`), in a per-night
  directory named by integer MJD: `<MJD>/trig_sn_young_<MJD>_targets.csv`, `_meta.yaml` and `<MJD>.log`. A rerun the
  same night overwrites. The run fails before querying if the output root does not exist.
- An empty target list is still written (header-only CSV): for a singular selector it deactivates yesterday's list.
- Logging goes only to the log file (DEBUG level), configured in `main` after parsing the CLI, not at import.
- Alerce queries are capped at `PAGE_SIZE = 100`; hitting the cap raises rather than silently truncating.
- TNS is rate limited; keep the `time.sleep(6)` between pages. The query uses the public search CSV with no API key
  (TNS returns 401 for an invalid key). The bot ID and name in the User-Agent are placeholders and must not be
  committed with real values. TNS RA/Dec arrive sexagesimal and are converted to decimal degrees in `query_tns`.
- Bandpass `lambda_eff` values in `trigger_output.KNOWN_BANDPASSES` come from the SVO Filter Profile Service; add new
  bands from there, never estimated.

## Git workflow (repo-specific)

While the project is in early development, the owner allows commits directly to `main` without pull requests. Still
make one commit per issue, and push only when asked.
- `*.csv`, `*.log`, `scratch/` and `.superpowers/` are git-ignored; the MJD output directories are run products.
