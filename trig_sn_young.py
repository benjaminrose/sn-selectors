#!/usr/bin/env -S uv run --script

# /// script
# requires-python = ">=3.12"
# dependencies = [
#                 "astropy",
#                 "alerce",
#                 "scipy",
#                 ]
# ///
"""
Built for cornjobs and saves logs along with output in folders by mjd.
"""

import argparse
import io
import logging
import operator
from pathlib import Path
import time

from alerce.core import Alerce
from astropy.coordinates import Angle, SkyCoord
from astropy.time import Time
import astropy.units as u
import numpy as np
import pandas as pd
import requests


ROOT_LOC = Path(__file__).parent.resolve()
NOW = Time.now()
log_file = ROOT_LOC / Path(f"{NOW.mjd:.0f}/{NOW.mjd:.0f}.log")
log_file.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=log_file, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CheckRange(argparse.Action):
    """https://gist.github.com/dmitriykovalev/2ab1aa33a8099ef2d514925d84aa89e7"""
    ops = {'inf': operator.gt,
           'min': operator.ge,
           'sup': operator.lt,
           'max': operator.le}

    def __init__(self, *args, **kwargs):
        if 'min' in kwargs and 'inf' in kwargs:
            raise ValueError('either min or inf, but not both')
        if 'max' in kwargs and 'sup' in kwargs:
            raise ValueError('either max or sup, but not both')

        for name in self.ops:
            if name in kwargs:
                setattr(self, name, kwargs.pop(name))

        super().__init__(*args, **kwargs)

    def interval(self):
        if hasattr(self, 'min'):
            l = f'[{self.min}'
        elif hasattr(self, 'inf'):
            l = f'({self.inf}'
        else:
            l = '(-infinity'

        if hasattr(self, 'max'):
            u = f'{self.max}]'
        elif hasattr(self, 'sup'):
            u = f'{self.sup})'
        else:
            u = '+infinity)'

        return f'valid range: {l}, {u}'

    def __call__(self, parser, namespace, values, option_string=None):
        for name, op in self.ops.items():
            if hasattr(self, name) and not op(values, getattr(self, name)):
                raise argparse.ArgumentError(self, self.interval())
        setattr(namespace, self.dest, values)


def convert_to_mjd(date_str):
    return Time(date_str, format="isot", scale="utc").mjd


def parse_cli():
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
        help="first detection was within AGE days",
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
        help="limiting magnitude used in slecting transients",
    )
    parser.add_argument(
        "-d",
        "--dec",
        default=-20,
        type=float,
        min=-90,
        sup=90,
        action=CheckRange,
        help="dec(deg) cut between apo and lco follow up",
    )
    args = parser.parse_args()
    logger.debug("parsed cli arguments")
    return args


def query_alerce(args, survey="ztf"):
    PAGE_SIZE = 100

    # Initialize the client
    alerce = Alerce()
    # print(alerce.query_classifiers())

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
        firstmjd=[NOW.mjd - args.age, NOW.mjd],
        index="oid",
        **params,
    )
    logger.info(f"found {len(ztf_objects)} from ZTF")
    if len(ztf_objects) >= PAGE_SIZE:
        logger.warning("Hit page size limit. Not all objects were downloaded.")
        raise RuntimeWarning("Hit page size limit. Not all objects were downloaded.")

    # lsst_objects = alerce.query_objects(survey="lsst", **params)
    # logger.info(f"Found {len(lsst_objects)} from LSST.")
    # if len(lsst_objects) >= PAGE_SIZE:
    # raise RuntimeWarning("Hit page size limit. Not all objects were downloaded.")

    objects = ztf_objects[["firstmjd", "meanra", "meandec", "probability"]]  # + lsst

    for obj in objects.index.values:
        detections = alerce.query_detections(oid=obj, format="pandas")
        detections = detections.sort_values(by="mjd")
        # fid 1 is g, fid 2 is r, fid 3 is i
        for i, flt in zip([1, 2, 3], ["g", "r", "i"]):
            try:
                objects.loc[obj, f"{flt}_mag"] = detections[
                    detections["fid"] == i
                ].iloc[0]["magpsf"]
            except IndexError:
                objects.loc[obj, f"{flt}_mag"] = None
    logger.debug("collected latest ztf photometry")
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
    API_KEY = "your_api_key"
    headers = {
        "User-Agent": f'tns_marker{{"tns_id":{TNS_BOT_ID},"type":"bot","name":"{TNS_BOT_NAME}"}}'
    }
    
    all_dfs = []
    page = 0
    logger.debug("requesting TNS data for ztf, atlas, and asas-sn")
    while True:
        url = f"{BASE_URL}&format=csv&num_page=500&page={page}"
        params = {"api_key": API_KEY}
        r = requests.get(url, headers=headers, params=params)

        if r.status_code != 200 or not r.text.strip():
            logger.warning(f"Stopping at page {page}, status {r.status_code}")
            logger.warning(f"Used {url=}, {headers=}, {params=}")
            break

        df = pd.read_csv(io.StringIO(r.text))
        if df.empty:
            break

        logger.info(f"page {page}: {len(df)} rows")
        all_dfs.append(df)
        page += 1
        time.sleep(6)  # be polite — TNS rate-limits this endpoint

    result = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"found {len(result)} from TNS")
    result = result[result["Obj. Type"].str.contains("SN", na=False)]
    logger.info(f"{len(result)} are typed as a SN")

    result = result.set_index("Name")
    result = result[
        [
            "RA",
            "DEC",
            "Obj. Type",
            "Redshift",
            "Discovery Date (UT)",
            "Discovery Filter",
            "Discovery Mag/Flux",
        ]
    ]
    result["discmjd"] = result["Discovery Date (UT)"].apply(
        lambda x: convert_to_mjd(x.replace(" ", "T"))
    )
    for filter in ["orange", "cyan", "u", "g", "r", "i", "z", "y"]:
        result[f"{filter}_mag"] = np.where(
            result["Discovery Filter"].str.contains(filter, na=False),
            result["Discovery Mag/Flux"],
            np.nan,
        )
    # from Tonry 2018
    result["probability"] = 1.0
    result = result.rename(
        columns={
            "RA": "ra",
            "DEC": "dec",
        }
    )
    result = result.drop(
        columns=["Discovery Date (UT)", "Discovery Filter", "Discovery Mag/Flux"]
    )
    return result


def deduplicate(objects):
    logger.debug(f"checking duplicates for {len(objects)}")
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

    objects = objects[keep].reset_index(drop=True)
    logger.debug(f"returning {len(objects)} de-duplicated targets")
    return objects


def clean_save(args, objects):
    # drop too faint
    mag_cols = [c for c in objects.columns if c.endswith("_mag")]
    mask = objects[mag_cols].lt(args.mag_lim).any(axis=1)
    objects = objects[mask]
    logger.info(f"{len(objects)} brighter than {args.mag_lim} mag")

    # sort by "priority" (probability * (YOUNG_DEF - age) -> normalized)
    objects["age"] = NOW.mjd - objects["discmjd"]
    objects["priority"] = objects["probability"] / objects["age"]
    objects["priority"] = objects["priority"] / objects["priority"].max()
    objects = objects.sort_values(by="priority", ascending=False)

    # add columns that are the same for all my objects
    # https://sdss-wiki.atlassian.net/wiki/spaces/DUE/pages/1837137928/TRIGGER+Selectors+output+format
    objects["program"] = ["sn_young"] * len(objects)
    objects["sky_brightness_mode"] = ["dark"] * len(objects)
    objects["n_exposures"] = [4] * len(objects)
    objects["active"] = [True] * len(objects)

    # split between two telescopes
    apo = objects[list(Angle(objects["dec"], unit="deg") > args.dec * u.arcsec)]
    lco = objects[list(Angle(objects["dec"], unit="deg") < args.dec * u.arcsec)]
    logger.info(f"sending {len(apo)} objects to APO and {len(lco)} objects to LCO")

    # save files
    for telescope, data in [("apo", apo), ("lco", lco)]:
        filepath = ROOT_LOC / Path(
            f"{NOW.mjd:.0f}/trig_sn_young_{telescope}_{NOW.mjd:.0f}.csv"
        )
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(filepath)
        logger.info(f"saved file {filepath}")


if __name__ == "__main__":
    args = parse_cli()

    # outputs should include columns
    # ['discmjd', 'ra', 'dec', 'probability', '*_mag']
    ztf = query_alerce(args)
    tns = query_tns(args)

    objects = pd.concat([ztf, tns], axis=0).sort_values(by="discmjd", ascending=False)
    objects = deduplicate(objects)
    clean_save(args, objects)
