"""Match the UI's run history (Laptop 2) against the uvicorn log (Laptop 1).

Usage: python -m scripts.match_requests ui_history.csv server.log
Every request ID shown in the UI must appear in the server log for the same trip.
"""
import argparse
import csv
import json
import re
from pathlib import Path


def server_requests(log_text):
    found = {}
    for line in log_text.splitlines():
        m = re.search(r"(\{.*\"request_id\".*\})", line)
        if m:
            rec = json.loads(m.group(1))
            found[rec["request_id"]] = rec
    return found


def main():
    p = argparse.ArgumentParser()
    p.add_argument("history", type=Path, help="ui_history.csv downloaded from the UI's Run history tab")
    p.add_argument("log", type=Path, help="uvicorn output saved on Laptop 1, e.g. server.log")
    a = p.parse_args()
    logged = server_requests(a.log.read_text(errors="replace"))
    rows = list(csv.DictReader(a.history.open()))
    ok = 0
    print(f"{'trip':<10}{'request id':<38}{'server log':<12}{'UTC time':<27}accuracy")
    for r in rows:
        rec = logged.get(r["request_id"])
        good = rec is not None and rec["trip_id"] == r["trip_id"]
        ok += good
        print(f"{r['trip_id']:<10}{r['request_id']:<38}{'MATCH' if good else 'MISSING':<12}"
              f"{(rec or {}).get('timestamp', '-'):<27}{(rec or {}).get('accuracy', '-')}")
    print(f"\n{ok}/{len(rows)} UI requests found in the server log for the same trip.")
    raise SystemExit(0 if rows and ok == len(rows) else 1)


if __name__ == "__main__":
    main()
