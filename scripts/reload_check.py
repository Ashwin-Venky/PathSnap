"""Compare persisted HTTP fingerprints with a fresh server process."""
import argparse
import hashlib
import json
import pandas as pd
import requests
from src.data import RESULT,save_json
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--url",required=True)
    args=parser.parse_args()
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
