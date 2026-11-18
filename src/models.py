import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from .data import CFG

MODELS={
    "Logistic":(LogisticRegression,dict(C=1.0,max_iter=300,random_state=CFG["seed"])),
    "GaussianNB":(GaussianNB,dict(var_smoothing=1e-7)),
    "DecisionTree":(DecisionTreeClassifier,dict(max_depth=12,min_samples_leaf=20,random_state=CFG["seed"])),
    "RandomForest":(RandomForestClassifier,dict(n_estimators=32,max_depth=14,min_samples_leaf=6,max_features="sqrt",n_jobs=2,random_state=CFG["seed"])),
    "HistBoost":(HistGradientBoostingClassifier,dict(max_iter=70,learning_rate=.10,max_leaf_nodes=23,l2_regularization=1.,early_stopping=False,random_state=CFG["seed"]))
}

def build(name,params):
    cls,defaults=MODELS[name]
    model=cls(**(defaults|params))
    return make_pipeline(StandardScaler(),model) if name in ("Logistic","GaussianNB") else make_pipeline(model)

def probabilities(model,ids,X):
    p=np.zeros(ids.shape,dtype=float)
    good=ids>=0
    if good.any(): p[good]=model.predict_proba(X[good])[:,1]
    return p

def search(trial,name):
    if name=="HistBoost":
        return dict(max_iter=trial.suggest_int("max_iter",35,110),max_leaf_nodes=trial.suggest_int("max_leaf_nodes",9,43),
                    learning_rate=trial.suggest_float("learning_rate",.035,.22,log=True),
                    l2_regularization=trial.suggest_float("l2_regularization",.01,12,log=True))
    if name=="RandomForest":
        return dict(n_estimators=trial.suggest_int("n_estimators",16,56),max_depth=trial.suggest_int("max_depth",7,21),
                    min_samples_leaf=trial.suggest_int("min_samples_leaf",2,30),max_features=trial.suggest_float("max_features",.3,.95))
    if name=="DecisionTree":
        return dict(max_depth=trial.suggest_int("max_depth",5,22),min_samples_leaf=trial.suggest_int("min_samples_leaf",3,80))
    if name=="Logistic": return dict(C=trial.suggest_float("C",.003,50,log=True))
    return dict(var_smoothing=trial.suggest_float("var_smoothing",1e-10,1e-2,log=True))
