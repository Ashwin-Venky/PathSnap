from contextlib import asynccontextmanager
from datetime import datetime,timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4
from collections import OrderedDict
import hashlib
from typing import Literal
import json
import logging
import os
import re
from threading import Lock
import numpy as np
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field,ConfigDict
from .data import ROOT,DATA,read_inputs
from .predict import predict,display_data,load_model,road_layer
from .private_score import score

Mode=Literal["development","evaluator"]
log=logging.getLogger("uvicorn.error")
class Request(BaseModel):
    model_config=ConfigDict(extra="forbid")
    mode:Mode
    trip_id:str=Field(pattern=r"^[0-9]{8}$")
class Result(BaseModel):
    mode:Mode
    request_id:str
    trip_id:str
    predicted_arc_ids:list[int]
    route_parts:list[list[int]]
    gps:list[list[float]]
    network:list[list[list[float]]]
    geometry:list[list[list[float]]]
    diagnostics:dict[str,int|float]
    metrics:dict[str,str|int|float|None]
    timing:dict[str,float]

@asynccontextmanager
async def lifespan(app):
    s=app.state
    tic=perf_counter()
    s.model=load_model()
    s.beta=json.loads((ROOT/"artifacts/decoder.json").read_text())["beta"]
    s.mode=os.getenv("PATHSNAP_MODE","development")
    if s.mode not in ("development","evaluator"): raise ValueError("Unknown server mode")
    dev=s.mode=="development"
    s.directory=DATA/"student_test" if dev else os.getenv("PATHSNAP_EVAL_INPUT")
    s.private=DATA/"student_test" if dev else os.getenv("PATHSNAP_PRIVATE")
    if not dev and s.private and s.directory and Path(s.private).resolve()==Path(s.directory).resolve():
        raise ValueError("Evaluator inputs and private references need separate directories")
    path=ROOT/"config/development.json" if dev else Path(os.getenv("PATHSNAP_EVAL_MANIFEST","/nonexistent"))
    s.trips=[]
    if path.is_file() and s.directory and Path(s.directory).is_dir():
        manifest=json.loads(path.read_text())
        trips=manifest["trips"]
        if manifest["mode"]!=s.mode or len(set(trips))!=10 or len(trips)!=10 or not all(re.fullmatch(r"[0-9]{8}",t) for t in trips):
            raise ValueError("Invalid trip manifest")
        s.trips=trips
    s.layers=OrderedDict()
    s.lock=Lock()
    s.load_s=perf_counter()-tic
    yield

app=FastAPI(title="PathSnap",lifespan=lifespan)
@app.get("/health")
def health():
    return dict(status="ready" if app.state.trips else "trips unavailable",mode=app.state.mode,model_load_s=app.state.load_s)
@app.get("/trips")
def trips():
    return dict(mode=app.state.mode,trips=app.state.trips)
@app.post("/predict",response_model=Result)
def run(req:Request):
    s=app.state
    if req.mode!=s.mode: raise HTTPException(409,"Requested mode differs from server mode")
    if req.trip_id not in s.trips: raise HTTPException(404,"Unknown trip")
    rid=str(uuid4())
    with s.lock:  # Bound concurrent model work on a demo laptop.
        tic=perf_counter()
        try:
            data=read_inputs(s.directory,req.trip_id)
            read_s=perf_counter()-tic
            ids,parts,diag,timing=predict(data,s.model,s.beta)
        except (ValueError,OSError):
            raise HTTPException(422,"Trip inputs could not be processed") from None
        ids.setflags(write=False)
        frozen=tuple(tuple(part) for part in parts)
        tic=perf_counter()
        try: metrics=score(s.mode,req.trip_id,s.private,data,ids,frozen)
        except (ValueError,OSError): metrics=dict(status="unavailable",reason="Private scoring input unreadable")
        timing.update(read_s=read_s,scoring_s=perf_counter()-tic)
        result=dict(request_id=rid,mode=s.mode,trip_id=req.trip_id,diagnostics=diag,metrics=metrics,timing=timing)
        log.info(json.dumps(dict(timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                 status="success",request_id=rid,trip_id=req.trip_id,mode=s.mode,
                                 observations=len(ids),inference_s=round(timing["inference_s"],4),
                                 scoring_s=round(timing["scoring_s"],4),scoring=metrics["status"],
                                 accuracy=round(metrics["accuracy"],4) if metrics.get("accuracy") is not None else None)))
        key=(hashlib.sha256(data.nodes).digest(),hashlib.sha256(data.arcs).digest())
        if key not in s.layers:
            s.layers[key]=road_layer(data)
            if len(s.layers)>2: s.layers.popitem(last=False)
        s.layers.move_to_end(key)
        return dict(**result,predicted_arc_ids=ids.tolist(),route_parts=parts,**display_data(data,parts,s.layers[key]))
