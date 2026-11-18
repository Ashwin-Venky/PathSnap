# PathSnap

PathSnap maps noisy vehicle observations to directed roads in a trip-specific network. A candidate classifier scores plausible segments. A log-space Viterbi decoder uses directed travel distance to reconstruct connected route pieces.

The local development run compared five models on 80 training trips, tuned histogram gradient boosting over 25 TPE trials, and exercised the real UI and API for all ten student-test trips. The consolidated technical report is [available here](reports/PathSnap_Technical_Report.pdf). All three teammate contributions are merged. A fresh checkout contains the complete implementation and fitted artifact. The report predates the selected v4 decoder; use the measurements below for the demo.

| Ten student-test trips, equal trip weight | Top-1 | Observation macro F1 | Directed continuity | Normalised edit error |
|---|---:|---:|---:|---:|
| Tuned classifier | 0.8912 | 0.7843 | 0.9087 | 0.5040 |
| Classifier and Viterbi | 0.8630 | 0.8129 | 0.9874 | 0.3829 |
| Selected v4 pure Viterbi, bounded projection | 0.9400 | 0.9058 | 0.9997 | 0.1085 |

Point scores use approximate route-derived labels for 18,773 of 19,160 student observations. The separate evaluator set scored 0.9351 equal-trip with 16,894 of 19,674 labels; these are not official organiser scores. Canonical point scoring and physical LAN verification remain pending.

## Selected one-laptop demo

Run both services from this project root in separate terminals. Keep private `.route` references in the external data directory; the API scores only after prediction. The active decoder is `artifacts/decoder.json`: beta 4, 25 m bounded projection correction, pure Viterbi. The fitted `artifacts/model.joblib` is unchanged.

```sh
export PATHSNAP_DATA="/path/to/directory/containing/student_test"
export OMP_NUM_THREADS=2
python -m uvicorn src.api:app --host 127.0.0.1 --port 18521 --access-log
```

In the second terminal, with the same environment installed:

```sh
export PATHSNAP_SERVER="http://127.0.0.1:18521"
python -m streamlit run ui.py --server.address 127.0.0.1 --server.port 18522 --browser.gatherUsageStats false
```

Open `http://127.0.0.1:18522` and select a student-test trip. Use Ctrl-C in each terminal to stop its service. The approximate evaluator `.route` score is supported server-side when the private route is present, and is explicitly labelled approximate; confirm any official scoring protocol with the organiser.

## Laptop 1 · model and API

Commands assume the project root. Use Python 3.13, preferably the tested 3.13.0. On macOS or Linux, create and activate the environment first.

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r config/server.txt
```

On Windows PowerShell use `py -3.13 -m venv .venv` and `.\.venv\Scripts\Activate.ps1`. Alternatively invoke `.\.venv\Scripts\python.exe` directly. All subsequent Python commands work in the project root on each OS.

Choose the external directory containing `train` and `student_test`. The default is the project's parent. The following macOS/Linux example creates it from the supplied ZIP in Downloads. The unpacker verifies existing files and preserves original row order.

```sh
export PATHSNAP_DATA="$HOME/PathSnap-data"
python -m scripts.unpack "$HOME/Downloads/Student folder.zip" "$PATHSNAP_DATA"
export OMP_NUM_THREADS=2
python -m uvicorn src.api:app --host 0.0.0.0 --port 18521
```

In PowerShell use `$env:PATHSNAP_DATA = "$env:USERPROFILE\PathSnap-data"`, `$env:OMP_NUM_THREADS = "2"` and `python -m scripts.unpack "$env:USERPROFILE\Downloads\Student folder.zip" "$env:PATHSNAP_DATA"`. The supplied development manifest lists the ten permitted IDs. Validate the receiving service locally with `curl http://127.0.0.1:18521/health`. Keep its terminal visible. It logs UTC timestamp, request ID, trip, mode, count, timings, scoring method and aggregate accuracy.

## Laptop 2 · evaluator interface

Copy or clone the merged repository. Laptop 2 needs the UI, the bundled Leaflet files and the UI dependencies. The data and fitted model stay on Laptop 1. Create a separate virtual environment as above and install `python -m pip install -r config/ui.txt`.

Find the server's active LAN IPv4 address with `ipconfig getifaddr en0` on macOS Wi-Fi, `ipconfig` on Windows or `ip -4 addr` on Linux. Use the active interface if its name differs. Set the **actual address shown on Laptop 1** in the following command.

```sh
export PATHSNAP_SERVER="http://192.168.1.23:18521"
curl "$PATHSNAP_SERVER/health"
python -m streamlit run ui.py --server.address 127.0.0.1 --server.port 18522 --browser.gatherUsageStats false
```

The example address is illustrative. In PowerShell set `$env:PATHSNAP_SERVER = "http://192.168.1.23:18521"` and run `Invoke-RestMethod "$env:PATHSNAP_SERVER/health"`. Open the local Streamlit page on port 18522, select a trip, and press **Run prediction**. The Connection sidebar can change the API URL. Localhost refers to the laptop making the request. Streamlit sends HTTP from its Python process. Leaflet JS/CSS and vector layers render offline.

Both laptops need a network that permits peer-to-peer traffic. If local health succeeds on Laptop 1 and remote health times out, check the interface, port and server bind address. Permit inbound TCP 18521 on Laptop 1 for the private LAN. If campus Wi-Fi isolates clients, use a permitted hotspot or wired LAN and repeat the health check.

## Evaluation mode and scoring

Laptop 1 owns mode selection. Set these variables only when the organisers supply the reserved data. Inputs use `INPUT/00000123/00000123.{track,nodes,arcs}` for each supplied ID. Mount references separately. Running the manifest generator checks ten IDs and their permitted file types.

```sh
export PATHSNAP_MODE=evaluator
export PATHSNAP_EVAL_INPUT="/path/to/reserved-inputs"
export PATHSNAP_PRIVATE="/path/to/private-scoring"
export PATHSNAP_EVAL_MANIFEST="/path/outside/repo/evaluator.json"
python -m scripts.serving_manifest --mode evaluator --directory "$PATHSNAP_EVAL_INPUT" --output "$PATHSNAP_EVAL_MANIFEST"
python -m uvicorn src.api:app --host 0.0.0.0 --port 18521
```

The optional custom point adapter reads `PRIVATE/TRIP/TRIP.points` only after prediction. Its contract is exactly one row per observation, in order, containing zero-based observation index and original arc ID. A value of -1 is unlabelled. Confirm this format and metric definition with the organiser before using an official mapping. Adapt the private scorer if their protocol differs. Missing files yield an unavailable scoring status while the route is returned. Missing evaluator inputs yield an empty list and an unavailable health status. When private `.route` files are available in either mode, server-side scoring uses the same frozen approximate alignment, clearly labelled as approximate.

GET /health reports mode and cold model loading time. GET /trips returns manifest IDs. POST /predict accepts JSON such as `{"mode":"development","trip_id":"00000000"}`. Mode mismatch returns 409, unknown ID 404, malformed ID or extra fields 422. Results contain one predicted arc ID per observation (-1 means unmatched), connected arc lists and latitude/longitude geometry, full GPS, display network, aggregate metrics, diagnostics and timings. The inference function accepts observations/network plus immutable algorithm assets. A two-entry cache reuses drawing geometry keyed by network content. Every request rebuilds candidates, scores them and decodes the trip.

## Reproduce the experiment

Install all research dependencies with `python -m pip install -r config/research.txt`. The measured machine was an Apple M1 Pro MacBook Pro with ten CPU cores and 16 GB RAM. Runs used two numerical threads and two random-forest workers. Seeds, group cap, radii, folds, TPE startup and trial budget are in config/settings.json. Python and direct library versions are in experiments/runtime.json.

```sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python -m scripts.audit --archive "$HOME/Downloads/Student folder.zip"
python -m scripts.prepare --pilot
python -m scripts.slice
python -m scripts.experiment
python -m scripts.ablation
python -m scripts.quality
python -m scripts.serving_manifest
python -m scripts.figures
python -m unittest discover -s tests -v
```

For an independent timed reproduction start in a fresh copy with an empty experiments/runs directory and a new study database. Preserve the distributed results for comparison. For resumption with identical inputs and code, each prepared trip has source hashes, each completed CV fold has a table and predictions, and Optuna reopens its named SQLite study. After changing a configuration or source file use a fresh run directory or checkout. Reusing stale fold tables across changed settings would invalidate comparison. The method uses finite native features, no class reweighting and no learned calibration. Whole observation groups were sampled for fitting. All valid labelled observations, including candidate misses, were evaluated. References live outside the predictor.

For a standalone early plumbing check, run audit and then `python -m scripts.slice`. This geometry-only run is separately identified in first_phase.json. The final selected model is the joblib artifact, with verified digest, version pins and feature schema. Its small size fits normal Git storage.

## Capture integration evidence and rebuild the report

With both services running locally and Chrome installed, execute the following on the research machine.

```sh
python -m scripts.integration
python -m scripts.browser_check
```

Stop and restart **the API process** and run `python -m scripts.reload_check --url http://127.0.0.1:18521`. This compares ten new predictions and route hashes to the original HTTP run. A new request ID is required. For the receiving-server image, use a dedicated terminal running Uvicorn, issue the development requests, and capture only that window as figures/server_terminal.png. The checked-in screenshot came from an actual Terminal process. Browser checks capture the Streamlit UI and the Folium HTML files generated under maps/. Then run `python -m scripts.report`. The report has been rendered and inspected. Its tables use results in experiments/ and its images use actual runs.

### Real-device acceptance

1. GET /health from Laptop 2 over the actual LAN and confirm the intended mode.
2. Request every reserved ID through Streamlit and correlate each request ID with Laptop 1's visible terminal.
3. Check that map pieces stop at breaks and private references stay server-side. Record organiser-approved scoring method, coverage and status.
4. Capture both physical device screens, record latency and network setup, and regenerate the report after team review.

See [HANDOFF.md](HANDOFF.md) for ownership, file transfer, review commands and integration order. The concluding report page contains viva prompts. AI assistance supported project implementation, execution and drafting. Team members should record only the implementation changes, tests and analysis they personally perform.

## Final integration acceptance

Testing evidence is saved under experiments/acceptance/. To retain the original development outputs, use the output option on the acceptance scripts.

```sh
python -m scripts.integration --output experiments/acceptance
python -m scripts.browser_check --output experiments/acceptance
# Stop and restart the API before this comparison.
python -m scripts.reload_check --url http://127.0.0.1:18521 --output experiments/acceptance
```

The .points format is a project-defined adapter. The supplied organiser documents provide no evidence of this file format. Confirm their scoring contract before final evaluation.
