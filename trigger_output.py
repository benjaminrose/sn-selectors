"""Shared helpers for writing SDSS-VI TRIGGER selector outputs.

Implements the selector-independent parts of the "TRIGGER Selectors output
format" spec: required target columns and their dtypes, brightness classes,
validation, run paths and the per-run metadata yaml. Selectors build a
DataFrame of targets and hand it to `write_run`.
"""

import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# column -> numpy dtype name; strings stay as python objects
REQUIRED_DTYPES: dict[str, str] = {
    "sel_objid": "object",
    "ra": "float64",
    "dec": "float64",
    "merit": "float32",
    "n_exposures": "int32",
    "dark_only": "bool",
    "opt_brightness": "object",
    "ir_brightness": "object",
    "fiber_type": "object",
    "observe_from": "float32",
    "observe_until": "float32",
}

BRIGHTNESS_CLASSES = ("B", "I", "F")
FIBER_TYPES = ("BOSS", "APOGEE")
MAX_OBJID_LENGTH = 32

MAG_PREFIX = "sel_mag_"
MAG_ERR_PREFIX = "sel_mag_err_"
MAG_MJD_PREFIX = "sel_mag_mjd_"

UTAH_TZ = ZoneInfo("America/Denver")

# Effective wavelengths (WavelengthEff, converted from Angstrom to nm) from the
# SVO Filter Profile Service, https://svo2.cab.inta-csic.es/theory/fps/
# filter IDs Palomar/ZTF.{g,r,i} and Misc/Atlas.{orange,cyan}, fetched 2026-11-23.
# ZTF alert (magpsf) and ATLAS discovery photometry are AB magnitudes.
KNOWN_BANDPASSES: dict[str, dict[str, object]] = {
    "ztf_g": {
        "system": "AB",
        "mag_type": "psf",
        "lambda_eff": 474.6,
        "err_metric": "1sigma",
    },
    "ztf_r": {
        "system": "AB",
        "mag_type": "psf",
        "lambda_eff": 636.6,
        "err_metric": "1sigma",
    },
    "ztf_i": {
        "system": "AB",
        "mag_type": "psf",
        "lambda_eff": 782.9,
        "err_metric": "1sigma",
    },
    "atlas_o": {"system": "AB", "mag_type": "psf", "lambda_eff": 663.0},
    "atlas_c": {"system": "AB", "mag_type": "psf", "lambda_eff": 518.2},
}


@dataclass(frozen=True)
class RunPaths:
    """Locations of everything one selector run writes."""

    run_dir: Path
    targets: Path
    meta: Path
    log: Path


def mag_columns(df: pd.DataFrame) -> list[str]:
    """Return the ``sel_mag_{x}`` columns, excluding errors and MJDs.

    Parameters
    ----------
    df : pandas.DataFrame
        Any frame of targets.

    Returns
    -------
    list of str
        Magnitude column names in frame order.
    """
    return [
        c
        for c in df.columns
        if c.startswith(MAG_PREFIX)
        and not c.startswith((MAG_ERR_PREFIX, MAG_MJD_PREFIX))
    ]


def opt_brightness_class(mag: float) -> str:
    """Classify an optical magnitude into the spec's B/I/F classes.

    Parameters
    ----------
    mag : float
        Magnitude; NaN means no measurement.

    Returns
    -------
    str
        "B" if mag < 13, "I" if 13 <= mag <= 16, otherwise "F".
    """
    if np.isnan(mag) or mag > 16:
        return "F"
    if mag < 13:
        return "B"
    return "I"


def add_spec_defaults(
    targets: pd.DataFrame,
    *,
    n_exposures: int = 4,
    dark_only: bool = True,
    fiber_type: str = "BOSS",
    ir_brightness: str = "none",
) -> pd.DataFrame:
    """Add the required columns that are the same for every target.

    ``opt_brightness`` is set from the brightest ``sel_mag_{x}`` of each row.

    Parameters
    ----------
    targets : pandas.DataFrame
        Targets, possibly with ``sel_mag_{x}`` columns.
    n_exposures, dark_only, fiber_type, ir_brightness
        Values written to every row.

    Returns
    -------
    pandas.DataFrame
        A copy with the new columns.
    """
    out = targets.copy()
    out["n_exposures"] = n_exposures
    out["dark_only"] = dark_only
    out["fiber_type"] = fiber_type
    out["ir_brightness"] = ir_brightness
    mags = out[mag_columns(out)]
    brightest = (
        mags.min(axis=1) if mags.shape[1] else pd.Series(np.nan, index=out.index)
    )
    out["opt_brightness"] = [opt_brightness_class(m) for m in brightest]
    return out


def coerce_dtypes(targets: pd.DataFrame) -> pd.DataFrame:
    """Cast required columns to their spec dtypes and magnitudes to float32.

    Parameters
    ----------
    targets : pandas.DataFrame
        Targets with all required columns.

    Returns
    -------
    pandas.DataFrame
        A copy with spec dtypes.
    """
    out = targets.copy()
    for col, dtype in REQUIRED_DTYPES.items():
        if col in out.columns:
            out[col] = out[col].astype(np.dtype(dtype))
    for col in out.columns:
        if col.startswith(MAG_PREFIX):
            out[col] = out[col].astype("float32")
    return out


def validate_targets(targets: pd.DataFrame) -> None:
    """Check a targets frame against the spec.

    Parameters
    ----------
    targets : pandas.DataFrame
        Targets to check.

    Raises
    ------
    ValueError
        Listing every problem found.
    """
    problems = [
        f"missing required column {c}" for c in REQUIRED_DTYPES if c not in targets
    ]
    if problems:
        raise ValueError("; ".join(problems))

    merit = targets["merit"].astype(float)
    if not merit.between(0.0, 1.0).all():
        problems.append("merit outside [0, 1] or NaN")
    objid = targets["sel_objid"].astype(str)
    if (objid.str.len() > MAX_OBJID_LENGTH).any():
        problems.append(f"sel_objid longer than {MAX_OBJID_LENGTH} characters")
    if objid.duplicated().any():
        problems.append(
            f"duplicate sel_objid: {sorted(set(objid[objid.duplicated()]))}"
        )
    if not targets["opt_brightness"].isin(BRIGHTNESS_CLASSES).all():
        problems.append(f"opt_brightness not in {BRIGHTNESS_CLASSES}")
    if not targets["fiber_type"].isin(FIBER_TYPES).all():
        problems.append(f"fiber_type not in {FIBER_TYPES}")
    if (targets["observe_until"] < targets["observe_from"]).any():
        problems.append("observe_until before observe_from")
    for col in ("ra", "dec", "observe_from", "observe_until"):
        if targets[col].isna().any():
            problems.append(f"{col} has missing values")
    if problems:
        raise ValueError("; ".join(problems))


def run_paths(output_root: Path, selector: str, mjd: int) -> RunPaths:
    """Build the paths for one run, ``<output_root>/<mjd>/...``.

    Parameters
    ----------
    output_root : pathlib.Path
        Selector base directory (everything up to the MJD directory).
    selector : str
        Selector short name.
    mjd : int
        Integer MJD of the run.

    Returns
    -------
    RunPaths
        Run directory, targets CSV, metadata yaml and log file.
    """
    run_dir = Path(output_root) / f"{mjd}"
    stem = f"{selector}_{mjd}"
    return RunPaths(
        run_dir=run_dir,
        targets=run_dir / f"{stem}_targets.csv",
        meta=run_dir / f"{stem}_meta.yaml",
        log=run_dir / f"{mjd}.log",
    )


def git_tag(repo_dir: Path) -> str:
    """Describe the code version with ``git describe --tags --always --dirty``.

    Falls back to the commit hash when there is no tag (``--always``).

    Parameters
    ----------
    repo_dir : pathlib.Path
        Directory inside the selector's git checkout.

    Returns
    -------
    str
        The description, or "unknown" if git fails.
    """
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--always", "--dirty"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        logger.warning(f"git describe failed in {repo_dir}; tag set to unknown")
        return "unknown"
    return result.stdout.strip() or "unknown"


def utah_isotime(t: datetime | None = None) -> str:
    """Format a time as ISO 8601 in Utah local time (America/Denver).

    Parameters
    ----------
    t : datetime.datetime, optional
        Timezone-aware time; defaults to now.

    Returns
    -------
    str
        e.g. "2026-11-23T12:00:00-07:00".
    """
    t = datetime.now(UTAH_TZ) if t is None else t.astimezone(UTAH_TZ)
    return t.isoformat(timespec="seconds")


def bandpass_stanza(targets: pd.DataFrame) -> dict[str, dict[str, object]]:
    """Describe the bandpasses that appear in the ``sel_mag_{x}`` columns.

    Parameters
    ----------
    targets : pandas.DataFrame
        Targets frame.

    Returns
    -------
    dict
        ``{x: {system, mag_type, lambda_eff, ...}}``; unknown bands map to ``{}``.
    """
    stanza: dict[str, dict[str, object]] = {}
    for col in mag_columns(targets):
        band = col.removeprefix(MAG_PREFIX)
        if band not in KNOWN_BANDPASSES:
            logger.warning(f"no bandpass metadata for {band}")
        stanza[band] = dict(KNOWN_BANDPASSES.get(band, {}))
    return stanza


def write_run(
    targets: pd.DataFrame,
    paths: RunPaths,
    *,
    selector: str,
    repo: str,
    chpc_username: str,
    tag: str,
    time_started: str,
    cmd: str,
) -> None:
    """Validate targets, then write the targets CSV and metadata yaml.

    Nothing is written if validation fails.

    Parameters
    ----------
    targets : pandas.DataFrame
        Targets with all required columns.
    paths : RunPaths
        Output locations from `run_paths`.
    selector, repo, chpc_username, tag, time_started, cmd : str
        Metadata values for the ``selector_run`` stanza.

    Raises
    ------
    ValueError
        If the targets fail `validate_targets`.
    """
    try:
        targets = coerce_dtypes(targets)
        validate_targets(targets)
    except ValueError as err:
        logger.error(f"targets failed validation, nothing written: {err}")
        raise
    paths.run_dir.mkdir(exist_ok=True)
    targets.to_csv(paths.targets, index=False)
    logger.info(f"saved {len(targets)} targets to {paths.targets}")

    meta = {
        "selector_run": {
            "selector": selector,
            "repo": repo,
            "tag": tag,
            "time_started": time_started,
            "time_datafile": utah_isotime(),
            "chpc_username": chpc_username,
            "datafile": paths.targets.name,
            "cmd": cmd,
        },
        "bandpasses": bandpass_stanza(targets),
    }
    paths.meta.write_text(yaml.safe_dump(meta, sort_keys=False))
    logger.info(f"saved metadata to {paths.meta}")
