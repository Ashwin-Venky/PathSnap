"""Called only after immutable predictions exist. Return aggregates only."""
from pathlib import Path
import numpy as np
from .alignment import align
from .geometry import Geometry
from .metrics import point_metrics,edit_error

def score(mode,trip,directory,data,ids,parts):
    if directory is None: return dict(status="unavailable",reason="Private references unavailable")
    base=Path(directory)/trip/trip
    if mode=="evaluator":
        mapping=base.with_suffix(".points")
        if mapping.is_file():
            table=np.loadtxt(mapping,dtype=int,ndmin=2)
            if table.shape!=(len(ids),2) or not np.array_equal(table[:,0],np.arange(len(ids))):
                return dict(status="unavailable",reason="Custom point mapping incompatible")
            truth=table[:,1]
            if ((truth < -1)|(truth >= len(data.arcs))).any():
                return dict(status="unavailable",reason="Custom point mapping contains invalid IDs")
            return dict(status="custom point mapping (organiser confirmation pending)",**point_metrics(truth,ids),coverage=float(np.mean(truth>=0)))
    path=base.with_suffix(".route")
    if not path.is_file(): return dict(status="unavailable",reason="Private reference unavailable; organiser format pending confirmation")
    route=np.loadtxt(path,dtype=int,ndmin=1)
    truth=align(Geometry(data),route)[0]
    status="approximate reference-derived accuracy" if mode=="evaluator" else "approximate reference alignment"
    return dict(status=status,**point_metrics(truth,ids),
                coverage=float(np.mean(truth>=0)),edit_error=edit_error(route,parts))
