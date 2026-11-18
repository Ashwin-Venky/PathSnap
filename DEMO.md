# PathSnap two-laptop demo

**Laptop 1** runs the model and the FastAPI server. **Laptop 2** runs the Streamlit UI and calls Laptop 1 over the Wi-Fi.
The data and model stay on Laptop 1. Commands are for macOS (Terminal); Windows PowerShell equivalents are underneath where they differ.

Both laptops need: this `PathSnap-demo` folder, Python 3.13, and the **same Wi-Fi or hotspot**.
Check with `python3.13 --version`; if it's missing on a Mac, run `brew install python@3.13` (on Windows install 3.13 from python.org). Run `conda deactivate` first if your prompt shows `(base)`. Campus Wi-Fi often blocks laptop-to-laptop traffic; if step 3 fails, use a phone hotspot.

---

## 1. Laptop 1: model and API server

```sh
cd ~/PathSnap-demo
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r config/server.txt

# Unpack the organisers' data once (the ZIP stays where it is)
export PATHSNAP_DATA="$HOME/PathSnap-data"
python -m scripts.unpack "$HOME/Desktop/Student folder.zip" "$PATHSNAP_DATA"

# Find this laptop's Wi-Fi address. Write it down; Laptop 2 needs it.
ipconfig getifaddr en0

# Start the server. Keep this window open; the log is also saved to server.log
export OMP_NUM_THREADS=2
python -m uvicorn src.api:app --host 0.0.0.0 --port 18521 2>&1 | tee server.log
```

Windows (PowerShell): `py -3.13 -m venv .venv`, `.\.venv\Scripts\Activate.ps1`, `$env:PATHSNAP_DATA="$env:USERPROFILE\PathSnap-data"`, `$env:OMP_NUM_THREADS="2"`, find the IP with `ipconfig` (IPv4 Address of the Wi-Fi adapter), and run uvicorn with `| Tee-Object server.log`.

If macOS asks **"Allow incoming network connections?"** for Python, click **Allow**. On Windows allow Python on **Private networks**.

**Check locally** (second Terminal window on Laptop 1):

```sh
curl http://127.0.0.1:18521/health
# expected: {"status":"ready","mode":"development","model_load_s":...}
```

## 2. Laptop 2: UI

```sh
cd ~/PathSnap-demo
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r config/ui.txt

# Use Laptop 1's address from step 1 (192.168.1.23 is only an example)
export PATHSNAP_SERVER="http://192.168.1.23:18521"
```

Windows: `$env:PATHSNAP_SERVER="http://192.168.1.23:18521"`.

## 3. Test the connection from Laptop 2

```sh
curl "$PATHSNAP_SERVER/health"
curl "$PATHSNAP_SERVER/trips"
```

Windows: `Invoke-RestMethod "$env:PATHSNAP_SERVER/health"`.

Both must answer. `/trips` must list 10 trips: 00000000, 00000007, 00000009, 00000040, 00000052, 00000056, 00000065, 00000076, 00000077, 00000091.
Laptop 1's server window should print a `GET /health ... 200 OK` line from Laptop 2's address.

## 4. Start the UI on Laptop 2

```sh
python -m streamlit run ui.py --server.address 127.0.0.1 --server.port 18522 --browser.gatherUsageStats false --client.toolbarMode viewer
```

Open <http://127.0.0.1:18522>. The sidebar should show **● Connected** with Laptop 1's address. If not, the red box says what to fix.

## 5. Run every trip and check the map

For each of the 10 trips: pick it in **Trip**, press **Run prediction**, and check that the map draws a green route over grey roads.
Use the **Map layers** buttons to switch layers on and off (try hiding Raw GPS and Road network to see the route alone).

Expected results (from the rehearsal on 18 Nov 2026; tiny accuracy differences between machines are normal):

| Trip | GPS points | Top-1 accuracy | Route pieces |
|---|---:|---:|---:|
| 00000000 | 2,503 | 94.8% | 10 |
| 00000007 | 1,354 | 96.7% | 81 |
| 00000009 | 1,049 | 89.1% | 3 |
| 00000040 | 2,995 | 88.7% | 83 |
| 00000052 | 262 | 98.5% | 1 |
| 00000056 | 3,958 | 88.3% | 11 |
| 00000065 | 768 | 81.4% | 12 |
| 00000076 | 1,486 | 88.4% | 2 |
| 00000077 | 3,911 | 57.2% | 66 |
| 00000091 | 874 | 80.2% | 2 |

Trip 00000077 is the known hard case (dense town, many short roads), so low accuracy there is expected.

## 6. Match UI requests with the server log

1. On Laptop 2, open the **Run history** tab under the map and press **Download run history (CSV)**. Copy `ui_history.csv` to Laptop 1 (AirDrop, USB or chat).
2. On Laptop 1, stop the server with **Ctrl+C** (server.log is already saved), then:

```sh
python -m scripts.match_requests ui_history.csv server.log
# expected last line: 10/10 UI requests found in the server log for the same trip.
```

You can also check by eye: each result shows **Request ID** under the trip title, and Laptop 1's window prints a JSON line with the same `request_id`.

## 7. Screenshots to save

Save them in `evidence/` with these names:

| File | What |
|---|---|
| `laptop1_server_terminal.png` | Laptop 1's uvicorn window showing the 10 JSON request lines |
| `laptop1_ip.png` | the `ipconfig getifaddr en0` output |
| `laptop2_health.png` | the `curl $PATHSNAP_SERVER/health` output on Laptop 2 |
| `laptop2_trip_00000000.png` | UI with trip 00000000 results and map |
| `laptop2_trip_00000077.png` | UI with the hard trip |
| `laptop2_run_history.png` | Run history tab with all 10 rows |
| `match_requests.png` | Terminal showing `10/10 UI requests found` |

On a Mac: **Cmd+Shift+4**, then **Space**, then click a window. On Windows: **Win+Shift+S**.

## Troubleshooting

| Symptom | Fix |
|---|---|
| UI: "Can't reach the API server" | Wrong IP, or the server isn't running with `--host 0.0.0.0`. Re-run `ipconfig getifaddr en0` on Laptop 1 (the IP changes between networks). |
| `curl` works on Laptop 1 but times out from Laptop 2 | Firewall or network isolation. Allow Python in the firewall, or switch both laptops to a phone hotspot. |
| `/health` says `"trips unavailable"` | `PATHSNAP_DATA` isn't set in the server's window or doesn't contain `student_test/`. Set it and restart uvicorn. |
| Server won't start: "Install the recorded model runtime versions" | Laptop 1 has different library versions. Recreate `.venv` and install `config/server.txt` again. |
| Port 18521 already in use | An old server is still running. Close it, or on macOS: `lsof -ti :18521 \| xargs kill`. |

## What changed in this version

- `ui.py`: redesigned layout, connection panel with **Test connection**, dataset and trip always shown, a loading status with steps, specific error messages, metric cards with explanations, larger map, layer toggles, route-piece table, run history with CSV download.
- `src/maps.py`: route drawn on top of the GPS points with a white outline, start/end markers, route-break markers, bottom legend, and an optional `layers` argument (existing callers are unchanged).
- `scripts/match_requests.py`: new; matches the UI run history with the server log.
- The API, model and predictions are unchanged.
