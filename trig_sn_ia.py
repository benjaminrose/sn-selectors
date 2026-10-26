#!/usr/bin/env -S uv run --script

# /// script
# requires-python = ">=3.12"
# dependencies = [
#                 "requests",
#                 "pycryptodome",
#                 "rich",
#                 ]
# ///
from rich import print

from fastdb_client import FASTDBClient

fdb = FASTDBClient("production")
# result = fdb.post("spectrum/spectrawanted", {})
result = fdb.post(
    "/objectsearch/edp2_cut2", json={
        "just_objids": True,
        # "min_numdetections": 2, 
        # "maxmag_maxdetection": 20,
        #"mjd_now": 61336
    }
)
# result = fdb.post( "/getprocvers" )

print(result)
