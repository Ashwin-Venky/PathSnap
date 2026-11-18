"""Reference-free prediction. The model receives only numeric geometry/motion features."""
from importlib.metadata import version
import json
from time import perf_counter
import joblib
import numpy as np
from .data import ROOT,CFG,digest
from .geometry import Geometry,candidates,FEATURES
from .models import probabilities
from .decoder import RoadPaths,decode,reconstruct,corrected_candidates

def load_model():
    path=ROOT/"artifacts/model.joblib"
    meta=json.loads((ROOT/"artifacts/model.json").read_text())
    if digest(path)!=meta["sha256"] or meta["features"]!=FEATURES or meta["settings"]!=CFG:
        raise ValueError("Artifact fingerprint or feature schema mismatch")
    if any(version(s)!=v for s,v in meta["versions"].items()):
        raise ValueError("Install the recorded model runtime versions")
    return joblib.load(path)

def predict(data,model,beta,projection_tolerance_m=0):
    start=perf_counter()
    g=Geometry(data)
    c=candidates(g)
    roads=RoadPaths(g)
    prepared=perf_counter()
    p=probabilities(model,c.ids,c.X)
    scored=perf_counter()
    projected=corrected_candidates(g,c,projection_tolerance_m)
    s,cut=decode(g,projected,p,roads,beta)
    ids,parts,diag=reconstruct(g,projected,s,roads,cut)
    done=perf_counter()
    diag.update(expanded=int(c.X[:,0,-1].sum()),zero_length_excluded=len(g.length)-len(g.valid))
    return ids,parts,diag,dict(features_s=prepared-start,classifier_s=scored-prepared,decoder_s=done-scored,inference_s=done-start)

def road_layer(data):
    edges=np.unique(np.sort(data.arcs,axis=1),axis=0)
    return data.nodes[:,::-1][edges].tolist()

def display_data(data,parts,network=None):
    # Coordinate order is converted exactly once at this interface.
    ll=data.nodes[:,::-1]
    paths=[]
    for part in parts:
        arcs=data.arcs[part]
        if len(arcs)>1 and not np.array_equal(arcs[:-1,1],arcs[1:,0]): raise ValueError("Broken reconstructed geometry")
        paths.append(ll[np.r_[arcs[:,0],arcs[-1,1]]].tolist())
    return dict(gps=data.track[:,[1,0]].tolist(),network=road_layer(data) if network is None else network,geometry=paths)
