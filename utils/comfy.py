from pathlib import Path
import json
import os
import argparse


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
