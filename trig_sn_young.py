#!/usr/bin/env -S uv run --script

# /// script
# requires-python = ">=3.12"
# dependencies = [
#                 "astropy",
#                 "alerce",
#                 "numpy",
#                 "pandas",
#                 "pyyaml",
#                 "requests",
#                 "scipy",
#                 ]
# ///
"""
Select young SNe from Alerce and TNS for SDSS-VI TRIGGER follow-up.

Built for cron jobs. Each run writes a targets CSV, a metadata yaml and a log
to ``<output-root>/<MJD>/`` following the SDSS-VI "TRIGGER Selectors output
format" spec, via `trigger_output`.
"""

import argparse
import io
import logging
import operator
import sys
import time
from datetime import UTC
from pathlib import Path
from typing import ClassVar

import astropy.units as u
import numpy as np
import pandas as pd
import requests
from alerce.core import Alerce
from astropy.coordinates import SkyCoord
from astropy.time import Time

from trigger_output import (
    add_spec_defaults,
    git_tag,
    mag_columns,
    run_paths,
    utah_isotime,
    write_run,
)

SELECTOR = "trig_sn_young"
REPO_URL = "https://github.com/benjaminrose/sn-selectors"
CHPC_USERNAME = "u6079968"
DEFAULT_OUTPUT_ROOT = (
    Path("/uufs/chpc.utah.edu/common/home/sdss10/sdss6work/due/selectors")
    / CHPC_USERNAME
    / SELECTOR
)
# merit uses max(age, AGE_FLOOR) so one very young object cannot dominate
AGE_FLOOR = 0.5  # days
# Alerce ZTF filter ids
ZTF_FIDS = {1: "g", 2: "r", 3: "i"}
# TNS band names that differ from the short bandpass name we use
TNS_BAND_ALIASES = {"orange": "o", "cyan": "c"}
SHARED_COLUMNS = ["discmjd", "ra", "dec", "probability"]

logger = logging.getLogger(__name__)


def empty_shared_frame() -> pd.DataFrame:
    """An empty frame with the shared query schema and float64 columns."""
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in SHARED_COLUMNS})


class CheckRange(argparse.Action):
    """https://gist.github.com/dmitriykovalev/2ab1aa33a8099ef2d514925d84aa89e7"""

    ops: ClassVar[dict] = {
        "inf": operator.gt,
        "min": operator.ge,
        "sup": operator.lt,
        "max": operator.le,
    }

    def __init__(self, *args, **kwargs):
        if "min" in kwargs and "inf" in kwargs:
            raise ValueError("either min or inf, but not both")
        if "max" in kwargs and "sup" in kwargs:
            raise ValueError("either max or sup, but not both")

        for name in self.ops:
            if name in kwargs:
                setattr(self, name, kwargs.pop(name))

        super().__init__(*args, **kwargs)

    def interval(self):
        if hasattr(self, "min"):
            l = f"[{self.min}"
        elif hasattr(self, "inf"):
            l = f"({self.inf}"
        else:
            l = "(-infinity"

        if hasattr(self, "max"):
            u = f"{self.max}]"
        elif hasattr(self, "sup"):
            u = f"{self.sup})"
        else:
            u = "+infinity)"

        return f"valid range: {l}, {u}"

    def __call__(self, parser, namespace, values, option_string=None):
        for name, op in self.ops.items():
            if hasattr(self, name) and not op(values, getattr(self, name)):
                raise argparse.ArgumentError(self, self.interval())
        setattr(namespace, self.dest, values)


def convert_to_mjd(date_str):
    return Time(date_str, format="isot", scale="utc").mjd


def parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Parameters
    ----------
    argv : list of str, optional
        Arguments; defaults to ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Query for young SN found with ZTF and LSST.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-a",
        "--age",
        default=7,
        type=float,
        min=0.0,
        action=CheckRange,
        help="first detection was within AGE days; targets are observed until AGE days after discovery",
    )
    parser.add_argument(
        "-p",
        "--prob",
        default=0.7,
        type=float,
        min=0.0,
        sup=1.0,
        action=CheckRange,
        help="probability threshold for SN-like",
    )
    parser.add_argument(
        "-m",
        "--mag_lim",
        default=21,
        type=float,
        help="limiting magnitude used in selecting transients",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        type=Path,
        help="selector base directory; each run writes to OUTPUT_ROOT/<MJD>/",
    )
    return parser.parse_args(argv)


def setup_logging(log_file: Path) -> None:
    """Send DEBUG logging to ``log_file`` only.

    Parameters
    ----------
    log_file : pathlib.Path
        Log file; its directory must exist.
    """
    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )


def latest_detection_mags(detections: pd.DataFrame) -> dict[str, float]:
    """Magnitude triples from the most recent ZTF detection in each band.

    Parameters
    ----------
    detections : pandas.DataFrame
        Alerce detections with ``fid``, ``mjd``, ``magpsf`` and ``sigmapsf``.

    Returns
    -------
    dict
        ``sel_mag_ztf_{b}``, ``sel_mag_err_ztf_{b}``, ``sel_mag_mjd_ztf_{b}``
        for b in g, r, i; NaN where a band has no detection.
    """
    out: dict[str, float] = {}
    for fid, band in ZTF_FIDS.items():
        in_band = detections[detections["fid"] == fid]
        if len(in_band):
            latest = in_band.sort_values("mjd").iloc[-1]
            values = (latest["magpsf"], latest["sigmapsf"], latest["mjd"])
        else:
            values = (np.nan, np.nan, np.nan)
        for prefix, value in zip(["sel_mag_", "sel_mag_err_", "sel_mag_mjd_"], values):
            out[f"{prefix}ztf_{band}"] = float(value)
    return out


def tns_bandpass(filter_str: str) -> str:
    """Bandpass short name from a TNS "Discovery Filter" string.

    Parameters
    ----------
    filter_str : str
        e.g. "r-ZTF" or "orange-ATLAS".

    Returns
    -------
    str
        ``{system}_{band}`` lowercased, e.g. "ztf_r", "atlas_o"; "unknown_{band}"
        when no system is given. Only ``[a-z0-9_]`` characters.
    """
    band, _, system = str(filter_str).partition("-")
    band = band.strip().lower()
    band = TNS_BAND_ALIASES.get(band, band)
    system = system.strip().lower() or "unknown"
    name = f"{system}_{band}"
    return "".join(c if c.isalnum() else "_" for c in name)


def compute_merit(
    probability: pd.Series, age: pd.Series, age_floor: float = AGE_FLOOR
) -> pd.Series:
    """Merit as probability / max(age, age_floor), normalized to a maximum of 1.

    Parameters
    ----------
    probability : pandas.Series
        SN-like probability of each target.
    age : pandas.Series
        Days since discovery.
    age_floor : float, optional
        Smallest age used, in days.

    Returns
    -------
    pandas.Series
        Merit in [0, 1]; the youngest targets score highest.
    """
    raw = probability / age.clip(lower=age_floor)
    if len(raw) == 0:
        return raw
    return raw / raw.max()


def query_alerce(args, now_mjd: float, survey="ztf"):
    """Query Alerce for SN-like ZTF objects first detected within ``args.age`` days.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI (uses ``age`` and ``prob``).
    now_mjd : float
        MJD of the run start; the search window is ``[now_mjd - age, now_mjd]``.
    survey : str, optional
        Alerce survey name.

    Returns
    -------
    pandas.DataFrame
        Shared-schema frame indexed by ZTF oid.
    """
    PAGE_SIZE = 100

    # Initialize the client
    alerce = Alerce()

    params = {
        #'stamp_classifier', also works.
        "classifier": "stamp_classifier_2025_beta",
        "class_name": "SN",
        "probability": args.prob,
        "page_size": PAGE_SIZE,
        "format": "pandas",
    }

    logger.debug("requesting ZTF alerts from Alerce")
    ztf_objects = alerce.query_objects(
        survey=survey,
        order_by="firstmjd",
        firstmjd=[now_mjd - args.age, now_mjd],
        **params,
    )
    # alerce's index="oid" raises on an empty result, so index afterwards
    if len(ztf_objects) == 0:
        logger.warning("no objects from Alerce")
        return empty_shared_frame()
    ztf_objects = ztf_objects.set_index("oid")
    logger.info(f"found {len(ztf_objects)} from ZTF")
    if len(ztf_objects) >= PAGE_SIZE:
        logger.warning("Hit page size limit. Not all objects were downloaded.")
        raise RuntimeWarning("Hit page size limit. Not all objects were downloaded.")

    objects = ztf_objects[["firstmjd", "meanra", "meandec", "probability"]].copy()

    mags = {}
    for obj in objects.index.values:
        detections = alerce.query_detections(oid=obj, format="pandas")
        mags[obj] = latest_detection_mags(detections)
    logger.debug("collected latest ztf photometry")
    objects = objects.join(pd.DataFrame.from_dict(mags, orient="index"))
    objects = objects.rename(
        columns={
            "firstmjd": "discmjd",
            "meanra": "ra",
            "meandec": "dec",
        }
    )
    return objects


def query_tns(args):
    # Grab From TNS for ztf, atlas, and asas-sn.
    # 18 and 3 are atlas and asas-sn.
    # first group and additional groups have different formats in the url,
    # so hardcode ztf (48) as first group
    groups_beyond_ztf = {"atlas": 18, "asas-sn": 3}
    reporting_group_string = "&reporting_groupid%5B%5D=48"
    for id in groups_beyond_ztf.values():
        reporting_group_string = reporting_group_string + "%2C" + str(id)
    BASE_URL = f"https://www.wis-tns.org/search?discovered_period_value={args.age}&discovered_period_units=days{reporting_group_string}"
    TNS_BOT_ID = "your_bot_id"
    TNS_BOT_NAME = "your_bot_name"
    headers = {
        "User-Agent": f'tns_marker{{"tns_id":{TNS_BOT_ID},"type":"bot","name":"{TNS_BOT_NAME}"}}'
    }

    all_dfs = []
    page = 0
    logger.debug("requesting TNS data for ztf, atlas, and asas-sn")
    while True:
        url = f"{BASE_URL}&format=csv&num_page=500&page={page}"
        # The public search needs no key; TNS returns 401 for an invalid api_key.
        r = requests.get(url, headers=headers)

        if r.status_code != 200 or not r.text.strip():
            logger.warning(f"Stopping at page {page}, status {r.status_code}")
            logger.warning(f"Used {url=}, {headers=}")
            break

        df = pd.read_csv(io.StringIO(r.text))
        if df.empty:
            break

        logger.info(f"page {page}: {len(df)} rows")
        all_dfs.append(df)
        page += 1
        time.sleep(6)  # be polite — TNS rate-limits this endpoint

    if not all_dfs:
        logger.warning("no TNS results; continuing without TNS objects")
        return empty_shared_frame()

    result = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"found {len(result)} from TNS")
    # a page of untyped ATs reads "Obj. Type" as float, so cast before .str
    result = result[result["Obj. Type"].astype(str).str.contains("SN", na=False)]
    logger.info(f"{len(result)} are typed as a SN")

    result = result.set_index("Name")
    result = result[
        [
            "RA",
            "DEC",
            "Discovery Date (UT)",
            "Discovery Filter",
            "Discovery Mag/Flux",
        ]
    ].copy()
    result["discmjd"] = result["Discovery Date (UT)"].apply(
        lambda x: convert_to_mjd(x.replace(" ", "T"))
    )
    # "Discovery Filter" is "<band>-<system>", e.g. "orange-ATLAS" or "r-ZTF".
    # Only the discovery mag is known: no error, measured at discmjd.
    bandpass = result["Discovery Filter"].apply(tns_bandpass)
    for x in sorted(bandpass.unique()):
        in_band = bandpass == x
        result[f"sel_mag_{x}"] = np.where(in_band, result["Discovery Mag/Flux"], np.nan)
        result[f"sel_mag_err_{x}"] = np.nan
        result[f"sel_mag_mjd_{x}"] = np.where(in_band, result["discmjd"], np.nan)
    # from Tonry 2018
    result["probability"] = 1.0
    # TNS gives sexagesimal strings; convert to decimal degrees to match Alerce
    coords = SkyCoord(result["RA"], result["DEC"], unit=(u.hourangle, u.deg))
    result["ra"] = coords.ra.deg
    result["dec"] = coords.dec.deg
    result = result.drop(
        columns=[
            "RA",
            "DEC",
            "Discovery Date (UT)",
            "Discovery Filter",
            "Discovery Mag/Flux",
        ]
    )
    return result


def deduplicate(objects):
    logger.debug(f"checking duplicates for {len(objects)}")
    if len(objects) < 2:
        return objects  # sky matching needs at least two objects
    coords = SkyCoord(ra=objects["ra"].values, dec=objects["dec"].values, unit="deg")
    # nthneighbor=2 skips self-match
    idx, sep, _ = coords.match_to_catalog_sky(coords, nthneighbor=2)
    keep = np.ones(len(objects), dtype=bool)
    for i in range(len(objects)):
        if keep[i] and sep[i] < 1 * u.arcsec:
            keep[idx[i]] = False  # drop the match, keep the first found
            logger.info(f"dropped {objects.iloc[idx[i]]}")
            logger.info("as as duplicate to")
            logger.info(f"{objects.iloc[i]}")

    # keep the index: it holds the ZTF oid or TNS name of each object
    objects = objects[keep]
    logger.debug(f"returning {len(objects)} de-duplicated targets")
    return objects


def build_targets(
    objects: pd.DataFrame, args: argparse.Namespace, now_mjd: float
) -> pd.DataFrame:
    """Turn merged, de-duplicated objects into TRIGGER target rows.

    Parameters
    ----------
    objects : pandas.DataFrame
        Indexed by object name, with ``discmjd``, ``ra``, ``dec``,
        ``probability`` and ``sel_mag*`` columns.
    args : argparse.Namespace
        Parsed CLI (uses ``age`` and ``mag_lim``).
    now_mjd : float
        MJD of this run.

    Returns
    -------
    pandas.DataFrame
        Spec columns plus ``discmjd``, ``probability`` and ``age``, sorted by
        merit (highest first).
    """
    # drop too faint
    mask = objects[mag_columns(objects)].lt(args.mag_lim).any(axis=1)
    targets = objects[mask].copy()
    logger.info(f"{len(targets)} brighter than {args.mag_lim} mag")

    # observe until AGE days after discovery; drop any whose window has closed
    targets["observe_from"] = now_mjd
    targets["observe_until"] = targets["discmjd"] + args.age
    expired = targets["observe_until"] <= now_mjd
    if expired.any():
        logger.info(f"dropping {expired.sum()} objects older than {args.age} days")
    targets = targets[~expired]

    targets["age"] = now_mjd - targets["discmjd"]
    targets["merit"] = compute_merit(targets["probability"], targets["age"])
    targets["sel_objid"] = targets.index.astype(str)
    targets = targets.drop(columns=["Obj. Type", "Redshift"], errors="ignore")
    targets = add_spec_defaults(targets)
    return targets.sort_values(by="merit", ascending=False).reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    """Run the selector and write one TRIGGER run directory."""
    # read the clock once; every date in this run derives from the start time
    start = Time.now()
    time_started = utah_isotime(start.to_datetime(timezone=UTC))
    args = parse_cli(argv)
    if not args.output_root.is_dir():
        raise OSError(f"output root {args.output_root} does not exist")
    paths = run_paths(args.output_root, SELECTOR, int(start.mjd))
    paths.run_dir.mkdir(exist_ok=True)
    setup_logging(paths.log)
    logger.debug(f"parsed cli arguments {args}")

    ztf = query_alerce(args, start.mjd)
    tns = query_tns(args)
    objects = pd.concat([ztf, tns], axis=0).sort_values(by="discmjd", ascending=False)
    objects = deduplicate(objects)
    targets = build_targets(objects, args, start.mjd)

    write_run(
        targets,
        paths,
        selector=SELECTOR,
        repo=REPO_URL,
        chpc_username=CHPC_USERNAME,
        tag=git_tag(Path(__file__).resolve().parent),
        time_started=time_started,
        cmd=" ".join(sys.argv if argv is None else [sys.argv[0], *argv]),
    )


if __name__ == "__main__":
    main()
