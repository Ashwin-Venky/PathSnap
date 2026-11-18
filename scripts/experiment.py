"""Initial five-fold model comparison, persistent TPE study, then final refit."""
import json
import os
from time import perf_counter
import warnings
import platform
import hashlib
from importlib.metadata import version
import joblib
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from src.data import ROOT,RESULT,CACHE,CFG,save_json,digest
from src.geometry import FEATURES
from src.models import MODELS,build,probabilities,search
from src.metrics import classification,point_metrics,select

def read_cache(part):
    data={}
    for p in sorted(CACHE.glob(part+"_*.npz")):
        with np.load(p) as z:
            data[p.stem.split("_")[-1]]={key:z[key] for key in ["ids","X","truth"]}
    return data

class Runner:
    def __init__(self):
        self.train=read_cache("train")
        self.test=read_cache("student_test")
        assert len(self.train)==80 and len(self.test)==10 and self.train.keys().isdisjoint(self.test)
        groups=np.concatenate([np.repeat(t,len(d["truth"])) for t,d in self.train.items()])
        self.folds=[(sorted(set(groups[a])),sorted(set(groups[b]))) for a,b in GroupKFold(CFG["folds"]).split(groups,groups=groups)]
        assert all(not set(a)&set(b) for a,b in self.folds)
        save_json(RESULT/"folds.json",[dict(train=a,validation=b) for a,b in self.folds])
        self.sample={}
        for trip,d in self.train.items():
            eligible=np.flatnonzero(d["truth"]>=0)
            rng=np.random.default_rng(CFG["seed"]+int(trip))
            ix=np.sort(rng.choice(eligible,min(len(eligible),CFG["fit_groups_per_trip"]),replace=False))
            ids=d["ids"][ix]
            valid=ids>=0
            self.sample[trip]=(d["X"][ix][valid],(ids==d["truth"][ix,None])[valid])

    def fit(self,name,params,trips):
        model=build(name,params)
        X,y=(np.concatenate([self.sample[t][j] for t in trips]) for j in (0,1))
        with warnings.catch_warnings(record=True) as notes:
            warnings.simplefilter("always")
            tic=perf_counter()
            model.fit(X,y)
        return model,perf_counter()-tic,[str(w.message) for w in notes]

    def evaluate(self,name,params,stage,fold,model,data,trips,fit_s,notes,trial=False):
        rows=[]
        preds={}
        for t in trips:
            d=data[t]
            tic=perf_counter()
            p=probabilities(model,d["ids"],d["X"])
            elapsed=perf_counter()-tic
            pred=select(p,d["ids"])
            if trial:
                rows.append(point_metrics(d["truth"],pred)["macro_f1"])
            else:
                rows.append(dict(model=name,stage=stage,fold=fold,trip=t,fit_s=fit_s,inference_s=elapsed,
                        warnings=" | ".join(notes),**classification(d["truth"],d["ids"],p)))
                preds[t]=pred
        return rows,preds

    def run(self,name,params,label,trial=None):
        rows=[]
        scores=[]
        directory=RESULT/"runs"/label
        directory.mkdir(parents=True,exist_ok=True)
        for fold,(a,b) in enumerate(self.folds):
            file=directory/f"cv{fold}.csv"
            if trial is None and file.exists() and (directory/f"cv{fold}.npz").exists():
                rows.extend(pd.read_csv(file,dtype={"trip":str}).to_dict("records"))
                continue
            model,elapsed,notes=self.fit(name,params,a)
            r,p=self.evaluate(name,params,"cv",fold,model,self.train,b,elapsed,notes,trial is not None)
            if trial is not None: scores.extend(r)
            else:
                pd.DataFrame(r).to_csv(file,index=False)
                np.savez_compressed(directory/f"cv{fold}.npz",**p)
                rows.extend(r)
            if trial is not None: trial.set_user_attr(f"fold_{fold}",float(np.mean(r)))
        if trial is not None: return float(np.mean(scores))
        model,elapsed,notes=self.fit(name,params,sorted(self.train))
        for stage,data in [("student_test",self.test),("train_refit",self.train)]:
            r,p=self.evaluate(name,params,stage,-1,model,data,sorted(data),elapsed,notes)
            pd.DataFrame(r).to_csv(directory/(stage+".csv"),index=False)
            np.savez_compressed(directory/(stage+".npz"),**p)
            rows.extend(r)
        save_json(directory/"configuration.json",dict(model=name,parameters=MODELS[name][1]|params,fit_s=elapsed,
                 seed=CFG["seed"],warnings=notes,cap=CFG["fit_groups_per_trip"],status="complete"))
        return pd.DataFrame(rows),model

def main():
    runner=Runner()
    frames=[]
    for name in MODELS:
        rows,_=runner.run(name,{},name)
        frames.append(rows)
        print(name,rows.groupby("stage").macro_f1.mean().to_dict(),flush=True)
    records=pd.concat(frames)
    records.to_csv(RESULT/"models.csv",index=False)
    rank=records[records.stage=="cv"].groupby("model").agg(f1=("macro_f1","mean"),loss=("binary_log_loss","mean")).reset_index()
    winner=rank.sort_values(["f1","loss","model"],ascending=[False,True,True]).iloc[0].model
    signature=hashlib.sha256(''.join(digest(p) for p in [RESULT/"feature_identity.json",RESULT/"folds.json",ROOT/"src/models.py",ROOT/"src/metrics.py",ROOT/"scripts/experiment.py"]).encode()).hexdigest()[:12]
    study=optuna.create_study(study_name=winner+"-"+signature,storage="sqlite:///"+str(RESULT/"tuning.sqlite3"),
        direction="maximize",sampler=optuna.samplers.TPESampler(seed=CFG["seed"],n_startup_trials=CFG["startup_trials"]),load_if_exists=True)
    complete=optuna.trial.TrialState.COMPLETE
    study.optimize(lambda t:runner.run(winner,search(t,winner),"tuning",t),n_trials=max(0,CFG["trials"]-len(study.get_trials(states=[complete]))))
    study.trials_dataframe().to_csv(RESULT/"trials.csv",index=False)
    rows,model=runner.run(winner,study.best_params,"Tuned")
    rows.to_csv(RESULT/"tuned.csv",index=False)
    p=ROOT/"artifacts/model.joblib"
    joblib.dump(model,p,compress=3)
    save_json(ROOT/"artifacts/model.json",dict(model=winner,parameters=MODELS[winner][1]|study.best_params,features=FEATURES,
              settings=CFG,study=study.study_name,best_trial=study.best_trial.number,cv_f1=study.best_value,
              sha256=digest(p),python=platform.python_version(),versions={s:version(s) for s in ["scikit-learn","numpy","scipy","joblib","shapely","pyproj"]}))
    print("EXPORTED",winner,study.best_value,flush=True)
if __name__=="__main__":
    with threadpool_limits(limits=2): main()
