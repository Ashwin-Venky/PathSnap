"""Read only permitted inference inputs. IDs always refer to original file rows."""
from pathlib import Path
from dataclasses import dataclass
import hashlib
import json
import os
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("PATHSNAP_DATA", ROOT.parent))
CACHE = ROOT / ".cache"
RESULT = ROOT / "experiments"
CFG = json.loads((ROOT / "config/settings.json").read_text())

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)

def trip_ids(directory):
    return sorted(p.name for p in Path(directory).glob("[0-9]" * 8) if p.is_dir())

@dataclass
class Inputs:
    track: np.ndarray
    nodes: np.ndarray
    arcs: np.ndarray

def read_inputs(directory, trip):
    base = Path(directory) / trip / trip
    track = np.loadtxt(base.with_suffix(".track"), ndmin=2)
    nodes = np.loadtxt(base.with_suffix(".nodes"), ndmin=2)
    arcs = np.loadtxt(base.with_suffix(".arcs"), dtype=np.int64, ndmin=2)
    if track.shape[1] != 3 or nodes.shape[1] != 2 or arcs.shape[1] != 2:
        raise ValueError("Unexpected file columns")
    for coords in (nodes, track[:, :2]):
        if not np.isfinite(coords).all() or (np.abs(coords) > [180, 90]).any():
            raise ValueError("Invalid coordinates")
    if not np.isfinite(track[:, 2]).all() or (np.diff(track[:, 2]) <= 0).any():
        raise ValueError("Invalid observation time")
    if (arcs < 0).any() or (arcs >= len(nodes)).any():
        raise ValueError("Invalid arc endpoint")
    return Inputs(track, nodes, arcs)
