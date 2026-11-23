# SDSS VI Supernova Selectors

Transient selectors developed by Ben Rose.

## trig_sn_young

A selector that uses the stamp classification from Alerce, plus TNS reports from ZTF, ATLAS and ASAS-SN, to find
young SN-like objects for SDSS-VI TRIGGER follow-up.

Each nightly run writes `<output-root>/<MJD>/trig_sn_young_<MJD>_targets.csv`, a matching `_meta.yaml` and a log,
in the SDSS-VI "TRIGGER Selectors output format". `--output-root` defaults to the selector's directory on Utah CHPC.

```bash
./trig_sn_young.py --help
./trig_sn_young.py -a 7 -p 0.7 -m 21 --output-root scratch/out
```

## trig_sn_ia

A selector to work with FastDB and select objects that are labeled as [`spectrum/spectrawanted`](https://fastdb.readthedocs.io/en/latest/usage.html#spectrum-endpoints). Not implemented yet. 

## trig_sn_host_redshift

A fiber filler to get host galaxies (for redshift and other physical properties) of cosmological SNe Ia. Not implemented yet. 
