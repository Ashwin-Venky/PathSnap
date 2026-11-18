"""Quantify geometry checks, alignment sensitivity and actual fitting group counts."""
import numpy as np
import pandas as pd
from pyproj import Geod
from src.data import *
from src.geometry import Geometry
from src.alignment import align

def main():
    rows=[]
    training_groups=training_rows=positive=0
    for part in ("train","student_test"):
        for trip in trip_ids(DATA/part):
            g=Geometry(read_inputs(DATA/part,trip))
            with np.load(CACHE/(part+"_"+trip+".npz")) as z:
                dist=z["distance"]
                truth=z["truth"]
                ids=z["ids"]
                if part=="train":
                    eligible=np.flatnonzero(truth>=0)
                    take=np.random.default_rng(CFG["seed"]+int(trip)).choice(eligible,min(len(eligible),CFG["fit_groups_per_trip"]),replace=False)
                    training_groups+=len(take)
                    training_rows+=int((ids[take]>=0).sum())
                    positive+=int(((ids[take]==truth[take,None])&(ids[take]>=0)).sum())
                arc=g.valid[::max(1,len(g.valid)//500)]
                coords=g.data.nodes
                pair=g.data.arcs[arc]
                metre=Geod(ellps="WGS84").inv(*coords[pair[:,0]].T,*coords[pair[:,1]].T)[2]
                err=np.abs(metre-g.length[arc])/np.maximum(metre,.001)
                rows.append(dict(trip=trip,partition=part,n=len(truth),coverage25=np.mean(dist<=25),coverage50=np.mean(dist<=50),
                                 coverage100=np.mean(dist<=100),geodesic_checks=len(arc),max_relative_projection_error=err.max()))
    pd.DataFrame(rows).to_csv(RESULT/"quality.csv",index=False)
    save_json(RESULT/"fit_distribution.json",dict(groups=training_groups,rows=training_rows,positives=positive,negatives=training_rows-positive))
if __name__=="__main__": main()
