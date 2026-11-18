"""Exercise the HTTP contract on all ten trips, repeat one prediction and reject invalid requests."""
import argparse
from pathlib import Path
from time import perf_counter
import hashlib
import json
import requests
import pandas as pd
from src.data import RESULT,save_json

def verify(url):
    health=requests.get(url+"/health",timeout=5).json()
    assert health["status"]=="ready" and health["mode"]=="development"
    ids=requests.get(url+"/trips",timeout=5).json()["trips"]
    assert len(ids)==10
    rows=[]
    for trip in ids+ids[:1]:
        tic=perf_counter()
        resp=requests.post(url+"/predict",json=dict(trip_id=trip,mode="development"),timeout=180)
        resp.raise_for_status()
        r=resp.json()
        elapsed=perf_counter()-tic
        assert set(r)=={"request_id","mode","trip_id","predicted_arc_ids","route_parts","gps","network","geometry","diagnostics","metrics","timing"}
        assert len(r["predicted_arc_ids"])==len(r["gps"])==r["diagnostics"]["observations"]
        assert r["metrics"]["status"]=="approximate reference alignment" and 0<r["metrics"]["coverage"]<=1
        assert set(r["metrics"])=={"status","coverage","labelled","accuracy","macro_f1","edit_error"}
        fingerprint=hashlib.sha256(json.dumps([r["predicted_arc_ids"],r["route_parts"]]).encode()).hexdigest()
        rows.append(dict(trip=trip,request_id=r["request_id"],http_s=elapsed,http_status=resp.status_code,prediction_sha256=fingerprint,
                         **r["diagnostics"],**r["metrics"],**r["timing"]))
    assert rows[0]["prediction_sha256"]==rows[-1]["prediction_sha256"] and rows[0]["request_id"]!=rows[-1]["request_id"]
    for trip,mode,status in [("../train","development",422),("99999999","development",404),(ids[0],"evaluator",409)]:
        assert requests.post(url+"/predict",json=dict(trip_id=trip,mode=mode),timeout=5).status_code==status
    pd.DataFrame(rows).to_csv(RESULT/"http.csv",index=False)
    save_json(RESULT/"server_health.json",health)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--url",default="http://127.0.0.1:18521")
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    if args.output:
        RESULT=args.output
        RESULT.mkdir(parents=True,exist_ok=True)
    verify(args.url)
