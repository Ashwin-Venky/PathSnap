"""Trip-centred metre coordinates and an index over complete directed segments."""
from dataclasses import dataclass
import numpy as np
from pyproj import Transformer
from shapely import linestrings, points, STRtree
from .data import CFG

FEATURES = ["distance_m","projection_fraction","length_m","heading_cos","heading_sin",
            "heading_valid","step_m","dt_s","speed_mps","turn_cos","out_degree","in_degree",
            "relative_distance_m","expanded_radius"]

class Geometry:
    def __init__(self, data):
        self.data = data
        lon, lat = np.median(data.track[:,:2], axis=0)
        self.crs = f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m"
        transform = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
        self.xy = np.column_stack(transform.transform(*data.track[:,:2].T))
        self.nodes = np.column_stack(transform.transform(*data.nodes.T))
        self.start = self.nodes[data.arcs[:,0]]
        self.vector = self.nodes[data.arcs[:,1]] - self.start
        self.length = np.linalg.norm(self.vector,axis=1)
        self.valid = np.flatnonzero(self.length > 1e-6)
        self.dt = np.diff(data.track[:,2],prepend=data.track[0,2])
        self.delta = np.diff(self.xy,axis=0,prepend=self.xy[:1])
        self.step = np.linalg.norm(self.delta,axis=1)
        self.motion = self.heading()

    def heading(self):
        # Whole-trip inference: use a short displacement window inside each time block.
        motion = np.zeros_like(self.xy)
        for ix in np.split(np.arange(len(self.xy)), np.flatnonzero(self.dt>CFG["gap_s"])):
            if not len(ix): continue
            left = np.maximum(ix-2,ix[0])
            right = np.minimum(ix+2,ix[-1])
            v = self.xy[right] - self.xy[left]
            norm = np.linalg.norm(v,axis=1)
            good = norm >= CFG["heading_min_m"]
            motion[ix[good]] = v[good] / norm[good,None]
        return motion

    def project(self, obs, ids):
        diff = self.xy[obs] - self.start[ids]
        v = self.vector[ids]
        fraction = np.clip(np.sum(diff*v,axis=-1)/np.maximum(self.length[ids]**2,1e-12),0,1)
        distance = np.linalg.norm(diff - fraction[...,None]*v,axis=-1)
        return fraction, distance

@dataclass
class Candidates:
    ids: np.ndarray
    frac: np.ndarray
    X: np.ndarray

def candidates(g, radius=None, k=None):
    radius = CFG["candidate_radius_m"] if radius is None else radius
    k = CFG["candidate_k"] if k is None else k
    n = len(g.xy)
    ids = np.full((n,k),-1,dtype=np.int32)
    frac = np.zeros((n,k))
    distance = np.zeros((n,k))
    expanded = np.zeros(n,dtype=bool)
    tree = STRtree(linestrings(g.nodes[g.data.arcs[g.valid]]))
    pt = points(g.xy)
    for obs_index, r in ((None,radius), (True,CFG["expanded_radius_m"])):
        query = np.arange(n) if obs_index is None else np.flatnonzero(ids[:,0]<0)
        if obs_index is not None: expanded[query]=True
        if not len(query): continue
        qi, si = tree.query(pt[query],predicate="dwithin",distance=r)
        if not len(si): continue
        oi, ai = query[qi], g.valid[si]
        f, d = g.project(oi, ai)
        order = np.lexsort((ai,d,oi))
        oi, ai, f, d = oi[order], ai[order], f[order], d[order]
        rank = np.arange(len(oi)) - np.repeat(np.flatnonzero(np.r_[True,oi[1:]!=oi[:-1]]), np.unique(oi,return_counts=True)[1])
        use = rank < k
        pos = (oi[use],rank[use])
        ids[pos], frac[pos], distance[pos] = ai[use], f[use], d[use]
    safe = np.maximum(ids,0)
    unit = g.vector[safe]/np.maximum(g.length[safe,None],1e-9)
    heading = g.motion[:,None,:]
    cos = np.sum(unit*heading,axis=2)
    sin = heading[:,:,0]*unit[:,:,1]-heading[:,:,1]*unit[:,:,0]
    dt = np.where(g.dt<=CFG["gap_s"],g.dt,0)
    step = np.where(dt>0,g.step,0)
    turn = np.sum(g.motion*np.roll(g.motion,1,axis=0),axis=1)
    turn[dt==0]=0
    arcs=g.data.arcs
    out_degree=np.bincount(arcs[:,0],minlength=len(g.nodes))
    in_degree=np.bincount(arcs[:,1],minlength=len(g.nodes))
    columns=[distance,frac,g.length[safe],cos,sin,np.any(heading!=0,axis=2),step[:,None],
             dt[:,None],(step/np.maximum(dt,1e-6))[:,None],turn[:,None],out_degree[arcs[safe,0]],
             in_degree[arcs[safe,1]],distance-distance[:,:1],expanded[:,None]]
    X=np.stack(np.broadcast_arrays(*columns),axis=-1).astype("float32")
    return Candidates(ids,frac,X)
