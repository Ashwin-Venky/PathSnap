"""Produce EDA, offline Folium maps, elapsed-time profiles and aggregate tables."""
import json
from time import perf_counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.data import *
from src.geometry import Geometry
from src.alignment import align
from src.predict import predict,display_data,load_model
from src.maps import route_map

F=ROOT/"figures"
def chart(fig,name):
    fig.tight_layout()
    fig.savefig(F/(name+".png"),dpi=150,bbox_inches="tight")
    plt.close(fig)

def main():
    (ROOT/"maps").mkdir(exist_ok=True)
    F.mkdir(exist_ok=True)
    meta=json.loads((ROOT/"artifacts/decoder.json").read_text())
    model=load_model()
    inv=pd.read_csv(RESULT/"inventory.csv",dtype={"trip":str})
    profiles=[]
    for part,trip in [("train","00000002"),("student_test","00000076"),("train",inv.loc[inv.observations.idxmax(),"trip"]),("train",inv.loc[inv.nodes.idxmax(),"trip"])]:
        d=read_inputs(DATA/part,trip)
        ids,parts,diag,timing=predict(d,model,meta["beta"])
        profiles.append(dict(partition=part,trip=trip,nodes=len(d.nodes),arcs=len(d.arcs),**diag,**timing))
        if trip not in ("00000002","00000076"):continue
        route=np.loadtxt(DATA/part/trip/(trip+".route"),dtype=int)
        g=Geometry(d)
        truth,occ,residual,ambiguous,starts=align(g,route)
        breaks=np.flatnonzero(d.arcs[route[:-1],1]!=d.arcs[route[1:],0])+1
        marks=display_data(d,parts)
        refs=display_data(d,np.split(route,breaks))["geometry"]
        route_map(**dict(gps=marks["gps"],network=marks["network"],geometry=marks["geometry"],reference=refs,
                  ambiguous=d.track[ambiguous][:,[1,0]].tolist())).save(ROOT/"maps"/(part+".html"))
        fig,ax=plt.subplots(figsize=(6.7,3.2))
        from matplotlib.collections import LineCollection
        ax.add_collection(LineCollection(g.nodes[d.arcs]/1000,colors="#c3c7c4",linewidths=.35))
        ax.add_collection(LineCollection(g.nodes[d.arcs[route]]/1000,colors="#765195",linewidths=1.5))
        ax.scatter(*g.xy.T/1000,s=1,color="#bd5d21",label="GPS observations")
        for j,loc in enumerate(breaks):
            coords=g.nodes[[d.arcs[route[loc-1],1],d.arcs[route[loc],0]]]/1000
            ax.scatter(*coords.T,s=50,marker="x",color="#b42140",label="Reference break ends" if j==0 else None)
        lo,hi=g.xy.min(axis=0)/1000,g.xy.max(axis=0)/1000
        ax.set(xlim=(lo[0]-.25,hi[0]+.25),ylim=(lo[1]-.25,hi[1]+.25),xlabel="Local east (km)",ylabel="Local north (km)",
               title=f"{part} · {trip} · road network and observations")
        ax.set_aspect("equal",adjustable="box")
        ax.legend(fontsize=7)
        chart(fig,part+"_eda")
    pd.DataFrame(profiles).to_csv(RESULT/"profiles.csv",index=False)
    d=read_inputs(DATA/"train","00000090")
    g=Geometry(d)
    fig,(ax,bx)=plt.subplots(1,2,figsize=(7.5,2.6))
    ax.scatter(np.arange(1,len(g.dt)),g.dt[1:],s=3,color="#236d75")
    ax.set(yscale="log",xlabel="Observation index",ylabel="Interval (s)",title="Training · 00000090")
    groups=np.cumsum(g.dt>CFG["gap_s"])
    bx.scatter(*g.xy.T/1000,c=groups,s=4,cmap="viridis")
    bx.set(xlabel="East (km)",ylabel="North (km)",title="Blocks separated by gaps >60 s")
    chart(fig,"gap")
    results=pd.concat([pd.read_csv(RESULT/"models.csv"),pd.read_csv(RESULT/"tuned.csv").assign(model="Tuned")])
    summary=[]
    for (m,stage),r in results.groupby(["model","stage"]):
        tm=r.groupby("fold").macro_f1.mean()
        row=dict(model=m,stage=stage,cv_fold_sd=tm.std() if len(tm)>1 else 0,
                 **{k:float(r[k].mean()) for k in ["accuracy","macro_f1","miss_rate","coverage","candidate_accuracy","precision","recall","candidate_macro_f1","binary_log_loss","conditional_obs_log_loss","inference_s"]},
                 weighted_accuracy=np.average(r.accuracy,weights=r.labelled),weighted_macro_f1=np.average(r.macro_f1,weights=r.labelled),
                 labelled=int(r.labelled.sum()),observations=int(r.observations.sum()),candidate_rows=int(r.candidate_rows.sum()),
                 weighted_binary_log_loss=np.average(r.binary_log_loss,weights=r.candidate_rows),
                 candidate_coverage=r.hit_count.sum()/r.labelled.sum(),obs_loss_conditional=np.average(r.conditional_obs_log_loss,weights=r.hit_count),
                 fit_s=r.groupby("fold").fit_s.first().sum(),**{k:int(r[k].sum()) for k in ["tp","tn","fp","fn"]})
        summary.append(row)
    s=pd.DataFrame(summary)
    s.to_csv(RESULT/"summary.csv",index=False)
    compare=s[s.model!="Tuned"]
    fig,axs=plt.subplots(1,3,figsize=(9,3.0))
    names=["Logistic","GaussianNB","DecisionTree","RandomForest","HistBoost"]
    for ax,title,metrics,stage in zip(axs,["Grouped validation","Student testing","Student testing"],[["macro_f1"],["accuracy","macro_f1"],["binary_log_loss","conditional_obs_log_loss"]],["cv","student_test","student_test"]):
        v=compare[compare.stage==stage].set_index("model").loc[names,metrics]
        v.columns=[{"macro_f1":"Macro F1","accuracy":"Top-1","binary_log_loss":"Binary loss","conditional_obs_log_loss":"Conditional loss"}[col] for col in metrics]
        v.plot.bar(ax=ax,color=["#177768","#cd8452"],width=.8,rot=38,fontsize=7)
        ax.set(title=title,xlabel="")
        ax.legend(fontsize=6)
    chart(fig,"model_comparison")
    t=pd.read_csv(RESULT/"trials.csv")
    fig,ax=plt.subplots(figsize=(7,2.5))
    ax.scatter(t.number,t.value,color="#ad6941",s=17)
    ax.plot(t.number,t.value.cummax(),color="#147662",label="Best so far")
    ax.axvspan(-.5,4.5,color="#aaa",alpha=.17,label="Startup trials")
    ax.set(xlabel="Trial number",ylabel="CV macro F1")
    ax.legend(fontsize=8)
    chart(fig,"tuning")
    a=pd.read_csv(RESULT/"ablation.csv")
    z=[]
    for m,r in a.groupby("method"):
        z.append(dict(method=m,**{k:r[k].mean() for k in ["accuracy","macro_f1","continuity","edit_error","route_s"]},
                 labelled=int(r.labelled.sum()),observations=int(r.observations.sum()),coverage=r.labelled.sum()/r.observations.sum(),
                 unmatched_rate=r.unmatched.sum()/r.observations.sum(),gap_boundaries=int(r.gap_boundaries.sum()),transitions=int(r.transitions.sum()),
                 restarts=int(r.restarts.sum()),weighted_accuracy=np.average(r.accuracy,weights=r.labelled),
                 weighted_macro_f1=np.average(r.macro_f1,weights=r.labelled),weighted_continuity=r.feasible.sum()/r.transitions.sum()))
    pd.DataFrame(z).to_csv(RESULT/"route_summary.csv",index=False)
if __name__=="__main__": main()
