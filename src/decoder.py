"""Bounded directed shortest paths and log-space Viterbi on projected arcs."""
from functools import lru_cache
import heapq
import numpy as np
from .data import CFG
from .metrics import weights

class RoadPaths:
    def __init__(self,g):
        self.g=g
        self.adj={}
        for arc in g.valid:
            u,v=g.data.arcs[arc]
            self.adj.setdefault(int(u),[]).append((int(v),float(g.length[arc]),int(arc)))
        # Instance-local cache. Duplicate edges retain original IDs; first equal
        # distance edge in original file order wins deterministic ties.
        self.search=lru_cache(maxsize=2048)(self._search)

    def _search(self,source,limit):
        dist={source:0.}
        via={}
        queue=[(0.,source)]
        while queue:
            dx,u=heapq.heappop(queue)
            if dx!=dist[u]: continue
            for v,length,arc in self.adj.get(u,()):
                new=dx+length
                if new<=limit and new<dist.get(v,np.inf):
                    dist[v]=new
                    via[v]=(u,arc)
                    heapq.heappush(queue,(new,v))
        return dist,via

    def bridge(self,a,b,f1,f2,budget):
        if a==b and f2>=f1:
            return [] if (f2-f1)*self.g.length[a]<=budget else None
        u,v=self.g.data.arcs[a,1],self.g.data.arcs[b,0]
        dist,via=self.search(int(u),bucket(budget))
        if dist.get(v,np.inf)+(1-f1)*self.g.length[a]+f2*self.g.length[b]>budget: return None
        path=[]
        while v!=u:
            v,arc=via[v]
            path.append(arc)
        return path[::-1]

def bucket(budget):
    return int(128*2**max(0,np.ceil(np.log2(max(1,budget)/128))))

def travel_matrices(g,c,roads):
    """Compute path lengths in chunks to cap intermediate node-pair allocations."""
    for start in range(1,len(c.ids),128):
        end=min(len(c.ids),start+128)
        a=c.ids[start-1:end-1,:,None]
        b=c.ids[start:end,None,:]
        aa,bb=np.maximum(a,0),np.maximum(b,0)
        f1,f2=c.frac[start-1:end-1,:,None],c.frac[start:end,None,:]
        dt=g.dt[start:end,None,None]
        budget=CFG["gps_slack_m"]+CFG["speed_limit_mps"]*dt
        ok=(a>=0)&(b>=0)&(dt<=CFG["gap_s"])
        same=(a==b)&(f2>=f1)
        d=np.where(same,(f2-f1)*g.length[aa],(1-f1)*g.length[aa]+f2*g.length[bb])
        valid=ok & (d<=budget) & ~same
        ix,ax,bx=np.nonzero(valid)
        if len(ix):
            nodes=len(g.nodes)
            source=g.data.arcs[aa[ix,ax,0],1]
            target=g.data.arcs[bb[ix,0,bx],0]
            bins=np.array([bucket(x) for x in budget[:,0,0]])
            # Encode observed node pairs within this chunk, deduplicate before lookup.
            keys=(bins[ix].astype(np.int64)*nodes+source)*nodes+target
            un,back=np.unique(keys,return_inverse=True)
            distances=[]
            for key in un:
                u_and_limit,v=divmod(int(key),nodes)
                limit,u=divmod(u_and_limit,nodes)
                distances.append(0. if u==v else roads.search(u,limit)[0].get(v,np.inf))
            d[ix,ax,bx]+=np.asarray(distances)[back]
        d[~ok | (d>budget)]=np.inf
        yield start,d

def decode(g,c,p,roads,beta=16):
    n,k=c.ids.shape
    emission=np.log(np.maximum(weights(np.asarray(p,dtype=float),c.ids),1e-100))
    emission[c.ids<0]=-np.inf
    links=np.full((n,k),-1,dtype=np.int16)
    cut=np.ones(n,bool)
    last=np.zeros(n,int)
    prev=emission[0]
    for start,dist in travel_matrices(g,c,roads):
        disp=g.step[start:start+len(dist),None,None]
        dt=g.dt[start:start+len(dist),None,None]
        transition=-np.abs(dist-disp)/beta-np.maximum(0,dist-35*dt)/beta
        for j,matrix in enumerate(transition):
            i=start+j
            last[i-1]=np.argmax(prev)
            score=prev[:,None]+matrix
            if np.isfinite(score).any():
                links[i]=score.argmax(axis=0)
                prev=emission[i]+score.max(axis=0)
                prev-=prev.max()
                cut[i]=False
            else: prev=emission[i].copy()
    last[-1]=np.argmax(prev)
    state=last.copy()
    for i in range(n-2,-1,-1):
        if not cut[i+1]: state[i]=links[i+1,state[i+1]]
    return state,cut

def reconstruct(g,c,states,roads,cuts=None):
    ix=np.arange(len(states))
    ids=c.ids[ix,states]
    frac=c.frac[ix,states]
    parts=[]
    feasible=0
    restarts=0
    for i,b in enumerate(ids):
        if b<0: continue
        path=None
        boundary=i==0 or g.dt[i]>CFG["gap_s"]
        if not boundary and ids[i-1]>=0 and (cuts is None or not cuts[i]):
            path=roads.bridge(ids[i-1],b,frac[i-1],frac[i],CFG["gps_slack_m"]+CFG["speed_limit_mps"]*g.dt[i])
        if path is None:
            parts.append([int(b)])
            restarts+=int(not boundary)
        else:
            feasible+=1
            if path or ids[i-1]!=b: parts[-1].extend(path+[int(b)])
    gaps=int((g.dt>CFG["gap_s"]).sum())
    denom=len(ids)-1-gaps
    return ids,parts,dict(observations=len(ids),matched=int((ids>=0).sum()),unmatched=int((ids<0).sum()),
                          gap_boundaries=gaps,restarts=restarts,feasible=feasible,transitions=denom,
                          continuity=feasible/max(denom,1),pieces=len(parts))
