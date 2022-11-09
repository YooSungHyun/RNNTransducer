from pathlib import Path
import json


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle)
