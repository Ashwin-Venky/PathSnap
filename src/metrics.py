"""Scores use reference-labelled observations, including candidate misses."""
import numpy as np
from rapidfuzz.distance import Levenshtein

def point_metrics(truth,pred):
    mask=truth>=0
    t,p=truth[mask],np.asarray(pred)[mask]
    if not len(t): return dict(labelled=0,accuracy=None,macro_f1=None)
    # Score the union of true and predicted local arc IDs. -1 counts as an error,
    # but is a status rather than a road class.
    extent=max(t.max(),p.max())+1
    true=np.bincount(t,minlength=extent)
    proposed=np.bincount(p[p>=0],minlength=extent)
    hits=np.bincount(t[t==p],minlength=extent)
    seen=(true+proposed)>0
    return dict(labelled=len(t),accuracy=float(np.mean(t==p)),macro_f1=float(np.mean(2*hits[seen]/(true+proposed)[seen])))

def weights(p,ids):
    value=np.where(ids>=0,np.clip(p,1e-8,1-1e-8),0)
    return value/np.maximum(value.sum(axis=1,keepdims=True),1e-100)

def select(p,ids):
    return ids[np.arange(len(ids)),np.where(ids>=0,p,-np.inf).argmax(axis=1)]

def classification(truth,ids,p):
    good=truth>=0
    valid=good[:,None] & (ids>=0)
    t=(ids==truth[:,None])[valid]
    q=np.clip(p[valid].astype(float),1e-8,1-1e-8)
    b=q>=.5
    tn,fp,fn,tp=np.bincount(t.astype(int)*2+b.astype(int),minlength=4).tolist()
    hit=((ids==truth[:,None]) & good[:,None]).any(axis=1)
    w=weights(p,ids)
    true_w=np.where(ids==truth[:,None],w,0).sum(axis=1)
    div=lambda a,b: a/max(b,1)
    return dict(**point_metrics(truth,select(p,ids)),observations=len(truth),miss_rate=1-div(hit.sum(),good.sum()),
        coverage=good.mean(),candidate_rows=len(q),candidate_accuracy=div(tn+tp,len(q)),
        precision=div(tp,tp+fp),recall=div(tp,tp+fn),candidate_macro_f1=(div(2*tp,2*tp+fp+fn)+div(2*tn,2*tn+fp+fn))/2,
        binary_log_loss=float(-np.where(t,np.log(q),np.log1p(-q)).mean()),tn=tn,fp=fp,fn=fn,tp=tp,
        hit_count=int(hit.sum()),conditional_obs_log_loss=float(-np.log(true_w[hit]).mean()) if hit.any() else None)

def compress(ids):
    a=np.asarray(ids)
    return a[np.r_[True,a[1:]!=a[:-1]]].tolist() if len(a) else []

def edit_error(reference,parts):
    # Breaks retain boundaries. Collapse only adjacent repetitions inside a piece.
    predicted=[x for part in parts for x in compress(part)]
    ref=compress(reference)
    return Levenshtein.distance(ref,predicted)/max(len(ref),len(predicted),1)
