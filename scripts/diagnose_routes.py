import json,hashlib,subprocess
from pathlib import Path
import numpy as np,pandas as pd,optuna
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from src.data import *
from src.geometry import Geometry,candidates
from src.predict import load_model
from src.models import probabilities
from src.decoder import RoadPaths,decode,reconstruct,travel_matrices
from src.alignment import align
from src.metrics import point_metrics
out=ROOT/'experiments/acceptance';out.mkdir(exist_ok=True)
meta=json.loads((ROOT/'artifacts/model.json').read_text());model=load_model()
identity=json.loads((RESULT/'feature_identity.json').read_text())
assert all(digest(ROOT/('src/'+f if f in ['alignment.py','geometry.py'] else 'config/'+f if f=='settings.json' else 'experiments/'+f))==sha for f,sha in identity['sources'].items())
study=optuna.load_study(study_name=meta['study'],storage='sqlite:///'+str(RESULT/'tuning.sqlite3'))
assert len(study.trials)==25 and all(t.state.name=='COMPLETE' for t in study.trials)
assert study.best_trial.number==meta['best_trial'] and abs(study.best_value-meta['cv_f1'])<1e-12
assert all(meta['parameters'][k]==v for k,v in study.best_params.items())
models=pd.read_csv(RESULT/'models.csv');assert models.model.nunique()==5
for name in models.model.unique():
 assert len(models[(models.model==name)&(models.stage=='cv')])==80
saved=np.load(RESULT/'runs/Tuned/student_test.npz');rows=[]; issues={};
for trip in trip_ids(DATA/'student_test'):
 d=read_inputs(DATA/'student_test',trip);g=Geometry(d);c=candidates(g);roads=RoadPaths(g);p=probabilities(model,c.ids,c.X)
 raw=c.ids[np.arange(len(c.ids)),np.where(c.ids>=0,p,-np.inf).argmax(axis=1)]
 assert np.array_equal(saved[trip],raw)
 state,cut=decode(g,c,p,roads,32);ids,parts,diag=reconstruct(g,c,state,roads,cut)
 route=np.loadtxt(DATA/'student_test'/trip/(trip+'.route'),dtype=int);truth=align(g,route)[0]
 row=dict(trip=trip,**diag,**point_metrics(truth,ids));rows.append(row)
 if trip not in ['00000040','00000007','00000077']:continue
 graph=csr_matrix((np.ones(len(d.arcs)),d.arcs.T),shape=(len(d.nodes),len(d.nodes)));_,comp=connected_components(graph,connection='weak')
 all_dist=np.concatenate([x for _,x in travel_matrices(g,c,roads)])
 reasons=[];alt=0;backward=0;weak=0;disconnect=0;overbudget=0
 for i in np.flatnonzero(cut)[1:]:
  if g.dt[i]>CFG['gap_s']:continue
  finite=np.isfinite(all_dist[i-1]);alt+=int(finite.any())
  a,b=ids[i-1:i+1];f1=c.frac[i-1,state[i-1]];f2=c.frac[i,state[i]]
  backwards=a==b and f2<f1;backward+=int(backwards)
  weak+=int(comp[d.arcs[a,1]]!=comp[d.arcs[b,0]])
  relaxed=roads.bridge(a,b,f1,f2,10000)
  disconnect+=int(relaxed is None);overbudget+=int(relaxed is not None)
  if len(reasons)<4:reasons.append(dict(observation=int(i),a=int(a),b=int(b),same_arc_backwards=bool(backwards),step_m=float(g.step[i]),dt=float(g.dt[i]),candidate_transition_exists=bool(finite.any())))
 issues[trip]=dict(**row,cut_count=int(cut[1:].sum()),cuts_with_other_feasible_candidates=alt,same_arc_backwards=backward,different_weak_components=weak,no_directed_route_within_10km=disconnect,directed_route_requires_more_than_budget=overbudget,examples=reasons)
 from src.maps import route_map
 from src.predict import display_data
 marks=display_data(d,parts); route_map(marks['gps'],marks['network'],marks['geometry']).save(out/(trip+'.html'))
pd.DataFrame(rows).to_csv(out/'point_checks.csv',index=False)
save_json(out/'fragmentation.json',issues)
save_json(out/'artifact_checks.json',dict(sha256=meta['sha256'],model=meta['model'],study=meta['study'],trials=25,best_trial=study.best_trial.number,cv_f1=study.best_value,feature_hashes_match=True,all_ten_classifier_predictions_match=True,models=sorted(models.model.unique()),commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
print(json.dumps(issues,indent=2))
