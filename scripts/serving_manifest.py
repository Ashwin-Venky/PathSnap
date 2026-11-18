"""Enumerate permitted ID and input files, ignoring all scoring files."""
import argparse
from pathlib import Path
from src.data import trip_ids,save_json,ROOT,DATA

def create(directory,path,mode):
    trips=trip_ids(directory)
    for t in trips:
        for ext in ("track","nodes","arcs"):
            if not (directory/t/(t+"."+ext)).is_file(): raise ValueError(f"Missing {ext} for {t}")
    if len(trips)!=10: raise ValueError("Expected exactly ten trips in selected partition")
    save_json(path,dict(mode=mode,trips=trips))
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--directory",type=Path,default=DATA/"student_test")
    p.add_argument("--output",type=Path,default=ROOT/"config/development.json")
    p.add_argument("--mode",choices=["development","evaluator"],default="development")
    a=p.parse_args()
    create(a.directory,a.output,a.mode)
