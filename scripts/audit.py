"""Recompute raw counts, network anomalies and archive identity without extracting."""
from pathlib import Path
from zipfile import ZipFile
import hashlib
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from pyproj import Geod
from src.data import DATA, RESULT, read_inputs, trip_ids, digest, save_json, CFG

def audit(archive=None):
    files, rows = [], []
    geod = Geod(ellps="WGS84")
    for partition in ("train", "student_test"):
        for trip in trip_ids(DATA / partition):
            directory = DATA / partition / trip
            d = read_inputs(DATA / partition, trip)
            n, a, t = d.nodes, d.arcs, d.track
            route = np.loadtxt(directory / (trip + ".route"), dtype=int, ndmin=1)
            assert route.min() >= 0 and route.max() < len(a)
            for ext in ("track", "arcs", "nodes", "route"):
                p = directory / (trip + "." + ext)
                files.append(dict(partition=partition, trip=trip, file=p.name, size=p.stat().st_size, sha256=digest(p)))
            graph = csr_matrix((np.ones(len(a)), a.T), shape=(len(n),len(n)))
            dt = np.diff(t[:,2])
            arc_m = geod.inv(*n[a[:,0]].T, *n[a[:,1]].T)[2]
            step_m = geod.inv(*t[:-1,:2].T, *t[1:,:2].T)[2]
            keys = a[:,0]*len(n)+a[:,1]
            rows.append(dict(partition=partition, trip=trip, observations=len(t), nodes=len(n), arcs=len(a), route=len(route),
                missing_values=0, invalid_indices=0, repeated_gps=len(t)-len(np.unique(t[:,:2],axis=0)),
                duplicate_nodes=len(n)-len(np.unique(n,axis=0)), duplicate_arcs=len(a)-len(np.unique(keys)),
                reverse_arcs=int(np.isin(a[:,1]*len(n)+a[:,0], keys).sum()), repeated_route=len(route)-len(np.unique(route)),
                min_lon=t[:,0].min(), max_lon=t[:,0].max(), min_lat=t[:,1].min(), max_lat=t[:,1].max(),
                dt_min=dt.min(), dt_median=np.median(dt), dt_max=dt.max(), one_second=int((dt==1).sum()),
                gap_count=int((dt>CFG["gap_s"]).sum()), track_m=step_m[dt<=CFG["gap_s"]].sum(),
                network_m=arc_m.sum(), route_m=arc_m[route].sum(), zero_arcs=int((n[a[:,0]]==n[a[:,1]]).all(axis=1).sum()),
                weak=connected_components(graph,connection="weak",return_labels=False),
                strong=connected_components(graph,connection="strong",return_labels=False),
                isolated=len(n)-len(np.unique(a)), breaks=int((a[route[:-1],1]!=a[route[1:],0]).sum())))
    if archive:
        with ZipFile(archive) as z:
            for f in files:
                suffix = '/'.join([f['partition'],f['trip'],f['file']])
                names = [n for n in z.namelist() if n.endswith('/'+suffix) or n==suffix]
                assert len(names)==1 and hashlib.sha256(z.read(names[0])).hexdigest()==f["sha256"]
    pd.DataFrame(files).to_csv(RESULT/"manifest.csv",index=False)
    table = pd.DataFrame(rows)
    table.to_csv(RESULT/"inventory.csv",index=False)
    print(table.groupby("partition")[["observations","nodes","arcs","route","zero_arcs","breaks"]].sum())
    save_json(RESULT/"input_identity.json",dict(manifest_sha256=digest(RESULT/"manifest.csv"), archive_sha256=digest(archive) if archive else None))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path)
    audit(parser.parse_args().archive)
