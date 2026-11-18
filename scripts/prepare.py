"""Freeze alignment and candidate settings, then materialise immutable per-trip arrays."""
from time import perf_counter
import numpy as np
import pandas as pd
from src.data import CFG, DATA, CACHE, ROOT, RESULT, digest, save_json, trip_ids, read_inputs
from src.geometry import Geometry, candidates, FEATURES
from src.alignment import align

def load_trip(part,trip):
    g=Geometry(read_inputs(DATA/part,trip))
    route=np.loadtxt(DATA/part/trip/(trip+".route"),dtype=int,ndmin=1)
    return g,align(g,route)

def pilot():
    rows=[]
    ids=trip_ids(DATA/"train")
    for trip in ids[::16]:
        g,(truth,*_)=load_trip("train",trip)
        row=truth>=0
        for radius,k in [(25,4),(35,8),(50,12)]:
            tic=perf_counter()
            c=candidates(g,radius,k)
            rows.append(dict(trip=trip,radius=radius,k=k,seconds=perf_counter()-tic,
                recall=(c.ids[row]==truth[row,None]).any(axis=1).mean(),rows=(c.ids>=0).sum()))
    frame=pd.DataFrame(rows)
    frame.to_csv(RESULT/"candidate_pilot.csv",index=False)
    means=frame.groupby(["radius","k"]).mean(numeric_only=True).reset_index()
    # Smallest cap within half a percentage point of best measured recall.
    choice=means[means.recall>=means.recall.max()-0.005].sort_values("k").iloc[0]
    CFG.update(candidate_radius_m=int(choice.radius),candidate_k=int(choice.k))
    save_json(ROOT/"config/settings.json",CFG)
    print(means.to_string(index=False),flush=True)

def prepare():
    CACHE.mkdir(exist_ok=True)
    hashes={p.name:digest(p) for p in [ROOT/"config/settings.json",ROOT/"src/alignment.py",ROOT/"src/geometry.py",RESULT/"manifest.csv"]}
    rows=[]
    for part in ["train","student_test"]:
        for trip in trip_ids(DATA/part):
            path=CACHE/(part+"_"+trip+".npz")
            meta=path.with_suffix(".json")
            if meta.exists() and path.exists():
                import json
                old=json.loads(meta.read_text())
                if old["sources"]==hashes:
                    rows.append(old["row"])
                    continue
            tic=perf_counter()
            g,(truth,occ,dist,ambiguity,bounds)=load_trip(part,trip)
            aligned=perf_counter()
            c=candidates(g)
            good=truth>=0
            row=dict(partition=part,trip=trip,crs=g.crs,observations=len(truth),labelled=int(good.sum()),
                     distant=int((~good).sum()),ambiguous=int(ambiguity.sum()),blocks=int(bounds.sum()),
                     recall=float((c.ids[good]==truth[good,None]).any(axis=1).mean()),
                     expanded=int(c.X[:,0,-1].sum()),unmatched=int((c.ids[:,0]<0).sum()),
                     median_residual_m=float(np.median(dist)),p95_residual_m=float(np.quantile(dist,.95)),
                     load_align_s=aligned-tic,features_s=perf_counter()-aligned)
            np.savez(path,truth=truth,occ=occ,distance=dist,ambiguous=ambiguity,boundaries=bounds,
                     ids=c.ids,X=c.X,frac=c.frac)
            save_json(meta,dict(sources=hashes,row=row))
            rows.append(row)
    pd.DataFrame(rows).to_csv(RESULT/"alignment.csv",index=False)
    save_json(RESULT/"feature_identity.json",dict(features=FEATURES,sources=hashes))
if __name__=="__main__":
    import sys
    if "--pilot" in sys.argv: pilot()
    prepare()
