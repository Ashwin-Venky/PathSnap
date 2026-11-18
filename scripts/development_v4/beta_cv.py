"""Retune the decoder scale only on whole-trip training folds after correction."""
import argparse,json
from pathlib import Path
from time import perf_counter
import numpy as np,pandas as pd
from src.data import DATA,CACHE,CFG,read_inputs,trip_ids
from src.geometry import Geometry,Candidates
from src.models import build,probabilities
from src.decoder import RoadPaths,decode,corrected_candidates,reconstruct
from src.metrics import point_metrics,edit_error

ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--betas',default='1,2,4,6,8,16,32,64,128,256');ap.add_argument('--tolerance',type=float,default=25);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
betas=[float(x) for x in a.betas.split(',')];folds=json.loads(Path('experiments/folds.json').read_text());params=json.loads(Path('artifacts/model.json').read_text())['parameters']
data={}
for trip in trip_ids(DATA/'train'):
    with np.load(CACHE/('train_'+trip+'.npz')) as z:c=Candidates(z['ids'],z['frac'],z['X']);truth=z['truth']
    eligible=np.flatnonzero(truth>=0);rng=np.random.default_rng(CFG['seed']+int(trip));ix=np.sort(rng.choice(eligible,min(len(eligible),CFG['fit_groups_per_trip']),replace=False));valid=c.ids[ix]>=0
    data[trip]=dict(c=c,truth=truth,sample=(c.X[ix][valid],(c.ids[ix]==truth[ix,None])[valid]))
rows=[]
for fold,split in enumerate(folds):
    X=np.concatenate([data[t]['sample'][0] for t in split['train']]);y=np.concatenate([data[t]['sample'][1] for t in split['train']])
    model=build('HistBoost',params);model.fit(X,y)
    for trip in split['validation']:
        c=data[trip]['c'];truth=data[trip]['truth'];g=Geometry(read_inputs(DATA/'train',trip));p=probabilities(model,c.ids,c.X);roads=RoadPaths(g);adjusted=corrected_candidates(g,c,a.tolerance)
        route=np.loadtxt(DATA/'train'/trip/(trip+'.route'),dtype=int,ndmin=1)
        for beta in betas:
            tic=perf_counter();state,cut=decode(g,adjusted,p,roads,beta);decoder_s=perf_counter()-tic
            pred,parts,diag=reconstruct(g,adjusted,state,roads,cut);m=point_metrics(truth,pred)
            rows.append(dict(fold=fold,trip=trip,beta=beta,tolerance_m=a.tolerance,labelled=m['labelled'],accuracy=m['accuracy'],macro_f1=m['macro_f1'],edit_error=edit_error(route,parts),continuity=diag['continuity'],pieces=diag['pieces'],restarts=diag['restarts'],decoder_s=decoder_s))
        print(fold,trip,flush=True)
f=pd.DataFrame(rows);f.to_csv(a.output/'beta_cv.csv',index=False)
s=f.groupby('beta').agg(accuracy=('accuracy','mean'),macro_f1=('macro_f1','mean'),edit_error=('edit_error','mean'),continuity=('continuity','mean'),pieces=('pieces','sum'),restarts=('restarts','sum'),decoder_s=('decoder_s','mean')).reset_index();s.to_csv(a.output/'beta_cv_summary.csv',index=False)
print(s.to_string(index=False),flush=True)
