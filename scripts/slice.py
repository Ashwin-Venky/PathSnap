"""First-stage geometry-only path reconstruction on one development trip."""
from time import perf_counter
import numpy as np
from src.data import *
from src.geometry import Geometry,candidates
from src.decoder import RoadPaths,decode,reconstruct
from src.metrics import point_metrics,edit_error
from src.alignment import align
if __name__=="__main__":
    trip=trip_ids(DATA/"student_test")[0]
    tic=perf_counter()
    g=Geometry(read_inputs(DATA/"student_test",trip))
    c=candidates(g)
    r=RoadPaths(g)
    state,cut=decode(g,c,np.exp(-c.X[:,:,0]/15),r)
    ids,parts,d=reconstruct(g,c,state,r,cut)
    elapsed=perf_counter()-tic
    route=np.loadtxt(DATA/"student_test"/trip/(trip+".route"),dtype=int)
    truth=align(g,route)[0]
    save_json(RESULT/"first_phase.json",dict(trip=trip,partition="student_test",method="geometry-only Viterbi plumbing check",
              prediction_s=elapsed,**d,**point_metrics(truth,ids),edit_error=edit_error(route,parts)))
