"""Compare persisted HTTP fingerprints with a fresh server process."""
import argparse
from pathlib import Path
import hashlib
import json
import pandas as pd
import requests
from src.data import RESULT,save_json
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--url",required=True)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.output:
        RESULT=args.output
        RESULT.mkdir(parents=True,exist_ok=True)
    records=[]
    expected=pd.read_csv(RESULT/"http.csv",dtype={"trip":str}).drop_duplicates("trip")
    for row in expected.itertuples():
        r=requests.post(args.url+"/predict",json={"mode":"development","trip_id":row.trip},timeout=180)
        r.raise_for_status()
        body=r.json()
        sha=hashlib.sha256(json.dumps([body["predicted_arc_ids"],body["route_parts"]]).encode()).hexdigest()
        assert sha==row.prediction_sha256 and body["request_id"]!=row.request_id
        records.append(dict(trip=row.trip,request_id=body["request_id"],original_request_id=row.request_id,fingerprint=sha))
    save_json(RESULT/"reload.json",records)
