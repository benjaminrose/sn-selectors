"""Shared fixtures for the selector tests."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# the selectors are plain scripts next to the repo root, not a package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def good_targets() -> pd.DataFrame:
    """Two rows that satisfy every required TRIGGER column."""
    return pd.DataFrame(
        {
            "sel_objid": ["ZTF26aaaaaaa", "2026abc"],
            "ra": [10.0, 200.0],
            "dec": [5.0, -30.0],
            "merit": [1.0, 0.25],
            "n_exposures": [4, 4],
            "dark_only": [True, True],
            "opt_brightness": ["F", "I"],
            "ir_brightness": ["none", "none"],
            "fiber_type": ["BOSS", "BOSS"],
            "observe_from": [61368.4, 61368.4],
            "observe_until": [61370.0, 61372.0],
            "sel_mag_ztf_r": [18.0, 15.0],
            "sel_mag_err_ztf_r": [0.05, np.nan],
            "sel_mag_mjd_ztf_r": [61367.0, 61366.0],
        }
    )
