# Inspecting prepared training candidates

`export.py` makes a readable CSV view of the **training** `.cache/train_XXXXXXXX.npz` arrays. One row is one GPS observation paired with one candidate directed road arc. The exporter writes only real candidates (`ids >= 0`), not padded `-1` slots. A trip ID and original observation index identify the GPS reading; `candidate_rank` identifies its slot among up to 12 nearby arcs; `candidate_arc_id` is the original zero-based row of that trip's `.arcs` file. Arc IDs are local to a trip and are metadata, not model features. CSV order is trip ID, observation index, then candidate rank.

The organiser's **Assessment and Plagiarism Policy**, section 4, prohibits sharing derived datasets between teams. **Deliverables** requires a public repository containing data-processing source code, but does not expressly permit republication of derived candidate-level training data. Therefore the real preview and full CSV export are generated under ignored `clean_datasets/local/` and are **not committed** pending organiser permission. This folder contains the reproducible exporter and its schema; it does not contain evaluator data or student-test labels. Do not upload generated CSVs to a public branch without written organiser confirmation.

## Regenerate locally

From the repository root, after preparing the original training partition with `PATHSNAP_DATA` pointing to the external data root:

```sh
export PATHSNAP_DATA='/path/to/data-root'
python -m scripts.prepare
python -m clean_datasets.export --mode preview --cache-dir .cache --output-dir clean_datasets/local/preview
python -m clean_datasets.export --mode full --cache-dir .cache --output-dir clean_datasets/local/full --max-rows-per-file 200000
```

The first command uses existing preprocessing code and creates or reuses `.cache/train_*.npz` files. The exporter itself reads the NPZs without changing them, the model, or the scoring code. It accepts **only** filenames beginning `train_` and requires exactly 80 training files. `preview` selects five distinct states, where available, from trip `00000002`: a labelled point, an ambiguous alignment, an unlabelled point, a candidate miss, and a radius-expanded point. Overlapping states may yield fewer than five observations. `full` processes all 80 trips and splits files after 200,000 candidate rows by default. Use a clean output directory when changing export options so old part files cannot be mistaken for a new run. These are local analysis views, not training inputs or official organiser scores.

## How the arrays were prepared

`scripts.prepare` validates original `.track`, `.nodes`, `.arcs`, and training `.route` files, preserving zero-based observation and arc row IDs. `src.geometry.Geometry` converts each trip to a local metre projection, excludes zero-length arcs from candidate search, and computes motion and road geometry. Candidate search starts within 50 m, expands to 160 m only when none are found, then keeps up to 12 directed arcs. `src.alignment.align` matches each GPS observation to an **occurrence** in the ordered training reference route using distance, heading, and monotone progress. It leaves points more than 50 m from the chosen reference arc unlabelled. Intersections, parallel roads, stops, and GPS error can make this reference-derived assignment ambiguous. It is an approximation, not an official point-to-arc label. The exporter copies saved arrays; it does not recompute labels.

## Data dictionary

| Column | Meaning |
|---|---|
| `trip_id`, `observation_index` | Eight-digit trip identifier and zero-based row of its `.track` file. |
| `candidate_rank`, `candidate_arc_id` | Zero-based candidate slot and original directed `.arcs` row ID within this trip. |
| `candidate_projection_fraction` | Saved float64 position of the GPS projection along the candidate arc, from 0 to 1; metadata, not an extra model feature. |
| `distance_m` | Perpendicular GPS-to-candidate distance in metres. |
| `projection_fraction` | Float32 feature copy of projected position along the arc. |
| `length_m` | Candidate arc length in metres. |
| `heading_cos`, `heading_sin`, `heading_valid` | Relative motion-to-arc direction and whether the short displacement window supplied a usable heading. |
| `step_m`, `dt_s`, `speed_mps` | Distance and elapsed time since prior GPS observation, and their ratio; values across a gap over 60 seconds are reset. |
| `turn_cos` | Cosine between successive estimated motion headings. |
| `out_degree`, `in_degree` | Directed graph degree at the candidate arc's start and end node. |
| `relative_distance_m` | Candidate distance minus the nearest candidate distance at that observation. |
| `expanded_radius` | 1 when the observation required the wider candidate search, else 0. |
| `reference_arc_id` | Approximate aligned training-route arc ID, blank when excluded by the 50 m rule. |
| `reference_occurrence_index` | Position in the ordered training `.route` occurrence sequence; blank for unlabelled points. |
| `reference_distance_m` | Distance to the chosen reference occurrence in metres, including unlabelled observations. |
| `reference_ambiguous`, `alignment_boundary` | Saved alignment ambiguity flag and reference block boundary flag (0/1). |
| `reference_status` | `aligned`, `ambiguous`, `candidate_miss`, or `unlabelled`. Candidate miss means a valid approximate reference exists but is absent from all candidate slots. |
| `candidate_label` | 1 if this candidate equals the approximate reference arc, 0 if it does not, blank when reference is unlabelled. Candidate misses remain 0 for every emitted candidate row. |

Only the 14 columns from `distance_m` through `expanded_radius` are model features. All identity, projection metadata, reference, status, and label columns are excluded from the classifier. The exporter checks NPZ keys, shapes, feature count, and finite values before writing rows. It prints the candidate-row count for each mode.
