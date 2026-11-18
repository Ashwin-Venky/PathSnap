"""Offline approximate labels from route occurrences, independent of model candidates."""
import numpy as np
from .data import CFG

def align(g, route):
    route = np.asarray(route, dtype=int)
    if route.ndim!=1 or not len(route) or (route<0).any() or (route>=len(g.length)).any():
        raise ValueError("Invalid reference sequence")
    arcs=g.data.arcs[route]
    piece=np.r_[0,np.cumsum(arcs[:-1,1]!=arcs[1:,0])]
    start_m=np.r_[0.,np.cumsum(g.length[route][:-1])]
    usable=np.flatnonzero(g.length[route]>1e-6)
    n=len(g.xy)
    k=min(CFG["alignment_k"],len(usable))
    if k==0:
        return np.full(n,-1),np.full(n,-1),np.full(n,np.inf),np.zeros(n,bool),np.ones(n,bool)
    occ=np.empty((n,k),int)
    progress=np.empty((n,k))
    cost=np.empty((n,k))
    dist=np.empty((n,k))
    ids=route[usable]
    direction=g.vector[ids]/g.length[ids,None]
    # Bound memory independently of trip size.
    for batch in np.array_split(np.arange(n),max(1,int(np.ceil(n*len(ids)/250000)))):
        f,d=g.project(batch[:,None],ids[None,:])
        unary=d+4*(1-g.motion[batch]@direction.T)
        order=np.argsort(unary,axis=1,kind="stable")[:,:k]
        occ[batch]=usable[order]
        cost[batch]=np.take_along_axis(unary,order,axis=1)
        dist[batch]=np.take_along_axis(d,order,axis=1)
        progress[batch]=start_m[usable[order]]+np.take_along_axis(f,order,axis=1)*g.length[ids[order]]
    prev=cost[0].copy()
    links=np.full((n,k),-1,int)
    ends={}
    boundaries=np.zeros(n,bool)
    boundaries[0]=True
    for i in range(1,n):
        delta=progress[i][None,:]-progress[i-1][:,None]
        valid=(occ[i][None,:]>=occ[i-1][:,None]) & (piece[occ[i]][None,:]==piece[occ[i-1]][:,None])
        valid &= (delta>=-15) & (delta<=CFG["gps_slack_m"]+CFG["speed_limit_mps"]*g.dt[i])
        transition=prev[:,None]+0.4*np.abs(delta-g.step[i])
        transition[~valid]=np.inf
        if g.dt[i]>CFG["gap_s"] or not np.isfinite(transition).any():
            ends[i-1]=prev.argmin()
            boundaries[i]=True
            prev=cost[i].copy()
        else:
            links[i]=transition.argmin(axis=0)
            prev=cost[i]+transition.min(axis=0)
            prev-=prev.min()
    ends[n-1]=prev.argmin()
    state=np.zeros(n,int)
    for i in range(n-1,-1,-1):
        state[i]=ends[i] if i in ends else links[i+1,state[i+1]]
    row=np.arange(n)
    chosen=occ[row,state]
    distance=dist[row,state]
    truth=route[chosen].copy()
    ambiguous=np.any((route[occ]!=truth[:,None]) & (np.abs(cost-cost[row,state,None])<=2),axis=1)
    truth[distance>CFG["label_max_m"]]=-1
    return truth,chosen,distance,ambiguous,boundaries
