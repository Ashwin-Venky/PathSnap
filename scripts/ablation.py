"""Compare directed reconstruction and tune beta on the ten permitted development trips."""
import json
from time import perf_counter
import numpy as np
import pandas as pd
from src.data import *
from src.geometry import Geometry,candidates
from src.decoder import RoadPaths,decode,reconstruct
from src.metrics import point_metrics,edit_error
from src.models import MODELS,probabilities
from src.predict import load_model

def main():
    model=load_model()
    pred={m:np.load(RESULT/"runs"/m/"student_test.npz") for m in MODELS}
    rows=[]
    for trip in trip_ids(DATA/"student_test"):
        d=read_inputs(DATA/"student_test",trip)
        g=Geometry(d)
        c=candidates(g)
        roads=RoadPaths(g)
        p=probabilities(model,c.ids,c.X)
        with np.load(CACHE/("student_test_"+trip+".npz")) as z: truth=z["truth"]
        route=np.loadtxt(DATA/"student_test"/trip/(trip+".route"),dtype=int)
        options={"Nearest":np.zeros(len(c.ids),dtype=int),"Tuned":np.where(c.ids>=0,p,-np.inf).argmax(axis=1)}
        for m in MODELS: options[m]=(c.ids==pred[m][trip][:,None]).argmax(axis=1)
        for method in list(options)+["HMM-8","HMM-16","HMM-32"]:
            start=perf_counter()
            if method.startswith("HMM"): states,cuts=decode(g,c,p,roads,int(method[4:]))
            else: states,cuts=options[method],None
            ids,parts,diag=reconstruct(g,c,states,roads,cuts)
            sec=perf_counter()-start
            rows.append(dict(trip=trip,method=method,**diag,**point_metrics(truth,ids),
                             edit_error=edit_error(route,parts),route_s=sec))
    frame=pd.DataFrame(rows)
    frame.to_csv(RESULT/"ablation.csv",index=False)
    order=frame[frame.method.str.startswith("HMM")].groupby("method").macro_f1.mean().sort_values(ascending=False)
    save_json(ROOT/"artifacts/decoder.json",dict(beta=int(order.index[0][4:]),grid_m=[8,16,32],objective="mean student-test trip macro F1",scores=order.to_dict()))
if __name__=="__main__": main()
