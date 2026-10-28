#!/usr/bin/env -S uv run --script

# /// script
# requires-python = ">=3.12"
# dependencies = [
#                 "astropy",
#                 "alerce",
#                 "rich",
#                 ]
# ///
"""
Built for cornjobs and saves logs along with output in folders by mjd.
"""

import argparse
import operator
import logging
from pathlib import Path
import sys

from alerce.core import Alerce
from astropy.time import Time
from rich import print

ROOT_LOC = path(__file__).partent.resolve()
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


parser = argparse.ArgumentParser(
    description="Query for young SN found with ZTF and LSST.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-a", "--age",
    default=7,
    type=float, min=0.0, action=CheckRange,
    help="first detection was within AGE days",
)
parser.add_argument(
    "-p", "--prob",
    default=0.7,
    type=float, min=0.0, sup=1.0, action=CheckRange,
    help="probability threshold for SN-like",
)
parser.add_argument(
    "-m", "--mag_lim",
    default=21,
    type=float,
    help="limiting magnitude used in slecting transients",
)
parser.add_argument(
    "-d", "--dec",
    default=-20,
    type=float, min=-90, sup=90, action=CheckRange,
    help="dec(deg) cut between apo and lco follow up",
)
args = parser.parse_args()
logger.debug("parsed cli arguments")
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
    "format": "pandas"
}

ztf_objects = alerce.query_objects(
    survey="ztf",
    order_by="firstmjd",
    firstmjd=[NOW.mjd - args.age, NOW.mjd],
    index="oid",
    **params
)
logger.info(f"Found {len(ztf_objects)} from ZTF.")
if len(ztf_objects) >= PAGE_SIZE:
    logger.warning("Hit page size limit. Not all objects were downloaded.")
    raise RuntimeWarning("Hit page size limit. Not all objects were downloaded.")

# lsst_objects = alerce.query_objects(survey="lsst", **params)
# logger.info(f"Found {len(lsst_objects)} from LSST.")
# if len(lsst_objects) >= PAGE_SIZE:
    # raise RuntimeWarning("Hit page size limit. Not all objects were downloaded.")
    
objects = ztf_objects[['firstmjd', 'lastmjd', 'meanra', 'meandec', "probability"]] #+ lsst

for obj in objects.index.values:
    detections = alerce.query_detections(oid=obj, format="pandas")
    detections = detections.sort_values(by="mjd", ascending=False)
    # fid 1 is g, fid 2 is r, fid 3 is i
    for i, flt in zip([1, 2, 3], ["g", "r", "i"]):
        try:
            objects.loc[obj, f"{flt}_mag"] = detections[detections["fid"] == i].iloc[0]["magpsf"]
        except IndexError:
            objects.loc[obj, f"{flt}_mag"] = None

#drop too faint
objects = objects[list(objects["r_mag"]<args.mag_lim)]
logger.info(f"{len(objects)} brighter than {args.mag_lim}")

#sort by "priority" (probability * (YOUNG_DEF - age) -> normalized)
objects["age"] = NOW.mjd-objects["firstmjd"]
objects["priority"] = objects["probability"]/objects["age"]
objects["priority"] = objects["priority"]/objects["priority"].max()
objects = objects.sort_values(by="priority", ascending=False)

#update to match required outputs
#https://sdss-wiki.atlassian.net/wiki/spaces/DUE/pages/1837137928/TRIGGER+Selectors+output+format
objects["program"] = ["sn_young"]*len(objects)
objects["sky_brightness_mode"] = ["dark"]*len(objects)
objects["n_exposures"] = [4]*len(objects)
objects["active"] = [True]*len(objects)
objects.rename(columns={
    "meanra": "ra",
    "meandec": "dec",
})

#split between two telescopes
apo = objects[list(objects["meandec"]>args.dec)]
lco = objects[list(objects["meandec"]<args.dec)]
logger.info(f"sending {len(apo)} objects to APO and {len(lco)} objects to LCO")

#save files
for telescope, data in [("apo", apo), ("lco", lco)]:
    filepath = ROOT_LOC / Path(f"{NOW.mjd:.0f}/trig_sn_young_{telescope}_{NOW.mjd:.0f}.csv")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(filepath)
    logger.info(f"saved file {filepath}")
