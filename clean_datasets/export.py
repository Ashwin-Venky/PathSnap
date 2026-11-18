"""Deterministic, training-only view of prepared GPS/candidate arrays."""
import argparse
import csv
from pathlib import Path
import numpy as np

from src.geometry import FEATURES

REQUIRED = {"truth", "occ", "distance", "ambiguous", "boundaries", "ids", "X", "frac"}
META = [
    "trip_id", "observation_index", "candidate_rank", "candidate_arc_id",
    "candidate_projection_fraction",
]
LABEL = [
    "reference_arc_id", "reference_occurrence_index", "reference_distance_m",
    "reference_ambiguous", "alignment_boundary", "reference_status", "candidate_label",
]
HEADER = META + list(FEATURES) + LABEL


def load_trip(path):
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != REQUIRED:
            raise ValueError(f"Unexpected NPZ keys: {path}")
        arrays = {key: z[key] for key in REQUIRED}
    ids, X, frac = arrays["ids"], arrays["X"], arrays["frac"]
    if ids.ndim != 2 or X.shape != ids.shape + (len(FEATURES),) or frac.shape != ids.shape:
        raise ValueError(f"Candidate/feature shape mismatch: {path}")
    if any(arrays[key].shape != (len(ids),) for key in ("truth", "occ", "distance", "ambiguous", "boundaries")):
        raise ValueError(f"Observation shape mismatch: {path}")
    if not np.isfinite(X[ids >= 0]).all() or not np.isfinite(frac[ids >= 0]).all():
        raise ValueError(f"Nonfinite valid candidate values: {path}")
    return arrays


def sample_indices(a):
    """A few deterministic observations illustrating distinct alignment states."""
    truth, ids = a["truth"], a["ids"]
    masks = [
        truth >= 0,
        a["ambiguous"],
        truth < 0,
        (truth >= 0) & ~(ids == truth[:, None]).any(axis=1),
        (a["X"][:, :, -1] > 0).any(axis=1),
    ]
    chosen = {int(np.flatnonzero(mask)[0]) for mask in masks if mask.any()}
    return sorted(chosen)


def rows(trip_id, a, observations):
    ids, X, frac, truth = (a[key] for key in ("ids", "X", "frac", "truth"))
    for i in observations:
        has_reference = truth[i] >= 0
        hit = has_reference and bool((ids[i] == truth[i]).any())
        status = ("unlabelled" if not has_reference else "candidate_miss" if not hit
                  else "ambiguous" if a["ambiguous"][i] else "aligned")
        for rank, arc in enumerate(ids[i]):
            if arc < 0:  # -1 is a padded slot, not a road candidate.
                continue
            yield [
                trip_id, i, rank, int(arc), format(float(frac[i, rank]), ".17g"),
                *(format(float(value), ".9g") for value in X[i, rank]),
                int(truth[i]) if has_reference else "",
                int(a["occ"][i]) if has_reference else "",
                format(float(a["distance"][i]), ".17g"),
                int(a["ambiguous"][i]), int(a["boundaries"][i]), status,
                int(arc == truth[i]) if has_reference else "",
            ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("clean_datasets/local"))
    parser.add_argument("--mode", choices=("preview", "full"), required=True)
    parser.add_argument("--max-rows-per-file", type=int, default=200_000)
    args = parser.parse_args()
    if args.max_rows_per_file < 1:
        parser.error("--max-rows-per-file must be positive")
    cache = args.cache_dir.resolve()
    files = sorted(cache.glob("train_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].npz"))
    if len(files) != 80:
        raise ValueError(f"Expected 80 training NPZ files; found {len(files)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    handle = None
    count = 0
    part = 0
    try:
        for path in files:
            trip_id = path.stem.removeprefix("train_")
            a = load_trip(path)
            selected = sample_indices(a) if args.mode == "preview" and trip_id == "00000002" else (range(len(a["ids"])) if args.mode == "full" else ())
            for row in rows(trip_id, a, selected):
                if writer is None or (args.mode == "full" and count % args.max_rows_per_file == 0):
                    if handle is not None:
                        handle.close()
                    part += 1
                    name = "training_preview.csv" if args.mode == "preview" else f"training_part_{part:03}.csv"
                    handle = (args.output_dir / name).open("w", newline="", encoding="utf-8")
                    writer = csv.writer(handle, lineterminator="\n")
                    writer.writerow(HEADER)
                writer.writerow(row)
                count += 1
    finally:
        if handle is not None:
            handle.close()
    if count == 0:
        raise ValueError("No candidate rows exported")
    print(f"mode={args.mode} files={part} candidate_rows={count} output={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
