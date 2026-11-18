"""Build the single technical report from retained results and genuine screenshots."""
from pathlib import Path
import json
from xml.sax.saxutils import escape
import numpy as np
import pandas as pd
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Paragraph,Table,TableStyle
from src.data import ROOT,RESULT

W,H=A4
M=46
WIDTH=W-2*M
INK="#182D32"
GREEN="#126A5E"
MUTED="#50656A"
def read(name): return json.loads((RESULT/name).read_text())
def csv(name): return pd.read_csv(RESULT/name,dtype={"trip":str})
def number(v):return f"{int(v):,}"
def val(v): return f"{v:.3f}"
def pct(v): return f"{100*v:.1f}%"

class Report:
    def __init__(self):
        self.pdf=Canvas(str(ROOT/"reports/PathSnap_Technical_Report.pdf"),pagesize=A4)
        self.pdf.setTitle("PathSnap | Technical report")
        self.page=0
    def start(self,section,title,source):
        if self.page: self.pdf.showPage()
        self.page+=1
        self.source=source
        c=self.pdf
        c.setFillColor(HexColor(GREEN)); c.setFont("Helvetica-Bold",9)
        c.drawString(M,H-44,section.upper())
        c.setFillColor(HexColor(INK)); c.setFont("Helvetica-Bold",25)
        c.drawString(M,H-78,title)
        c.setStrokeColor(HexColor("#CEDCD8"));c.line(M,H-94,W-M,H-94)
        c.setFont("Helvetica",8);c.setFillColor(HexColor(MUTED))
        c.drawString(M,25,source);c.drawRightString(W-M,25,str(self.page))
        self.y=H-111
    def para(self,s,size=10.5):
        style=ParagraphStyle("body",fontName="Helvetica",fontSize=size,leading=size*1.39,textColor=HexColor(INK))
        p=Paragraph(s,style)
        _,h=p.wrap(WIDTH,1000)
        self.y-=h
        p.drawOn(self.pdf,M,self.y);self.y-=10
        self.check()
    def head(self,s):
        self.y-=14
        self.pdf.setFillColor(HexColor(GREEN));self.pdf.setFont("Helvetica-Bold",14)
        self.pdf.drawString(M,self.y,s);self.y-=9
    def table(self,rows,widths=None,size=8.6):
        t=Table(rows,colWidths=[WIDTH*x for x in widths] if widths else [WIDTH/len(rows[0])]*len(rows[0]))
        t.setStyle(TableStyle([("FONTNAME",(0,0),(-1,-1),"Helvetica"),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("TEXTCOLOR",(0,0),(-1,-1),HexColor(INK)),("FONTSIZE",(0,0),(-1,-1),size),
            ("BACKGROUND",(0,0),(-1,0),HexColor("#DCEBE6")),("ROWBACKGROUNDS",(0,1),(-1,-1),[HexColor("#F5F8F6"),HexColor("#FFFFFF")]),
            ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),4),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
            ("ALIGN",(1,1),(-1,-1),"RIGHT")]))
        _,h=t.wrap(WIDTH,1000)
        self.y-=h
        t.drawOn(self.pdf,M,self.y);self.y-=12
        self.check()
    def image(self,name,max_h,caption):
        path=ROOT/"figures"/(name+".png")
        pic=ImageReader(str(path));w,h=pic.getSize()
        scale=min(WIDTH/w,max_h/h)
        self.y-=h*scale
        self.pdf.drawImage(pic,M+(WIDTH-w*scale)/2,self.y,width=w*scale,height=h*scale)
        self.y-=6
        self.para(escape(caption),9)
    def check(self):
        if self.y<40: raise ValueError(f"Page {self.page} overflows at {self.y}")
    def finish(self): self.pdf.save()

def main():
    r=Report()
    inv=csv("inventory.csv"); lab=csv("alignment.csv")
    s=csv("summary.csv").set_index(["model","stage"])
    a=csv("route_summary.csv").set_index("method")
    ab=csv("ablation.csv"); conf=json.loads((ROOT/"artifacts/model.json").read_text())
    dconf=json.loads((ROOT/"artifacts/decoder.json").read_text()); dec="HMM-"+str(dconf["beta"])
    fit=read("fit_distribution.json"); http=csv("http.csv").drop_duplicates("trip")
    ui=read("ui_runs.json"); trials=csv("trials.csv")
    t=a.loc["Tuned"]; h=a.loc[dec]
    r.start("PES University  /  UE24CS352A Machine Learning","PathSnap","Development evidence  |  18 November 2026")
    r.para("Reconstructing a vehicle's route from noisy GPS observations on a directed road network",17)
    r.para("Technical report for the PathSnap hackathon. The development run uses 80 training trips and ten organiser-supplied student-test trips. Reserved evaluator trips were unavailable.")
    r.head("Measured outcome")
    r.table([["Ten student-test trips","Top-1","Macro F1","Continuity","Edit error"],["Tuned classifier",pct(t.accuracy),val(t.macro_f1),pct(t.continuity),val(t.edit_error)],["Classifier + Viterbi",pct(h.accuracy),val(h.macro_f1),pct(h.continuity),val(h.edit_error)]],[.31,.16,.16,.18,.19])
    r.para(f"The decoder raised continuity and trip-local macro F1 while reducing point accuracy. Point metrics cover {number(h.labelled)} of {number(h.observations)} observations ({pct(h.coverage)}), using approximate labels derived from the supplied routes. Each trip has equal weight in this table.")
    r.head("What was exercised")
    r.para(f"Five classifiers completed five held-out trip folds. Histogram gradient boosting won the initial comparison. A persistent TPE study completed {len(trials)} trials. Every development trip ran through the HTTP API and the Streamlit map. Reloaded predictions matched by SHA-256 for all ten trips.")
    r.head("Submission status")
    r.para("Development report for team review. The same computer hosted the client and server during these tests. The physical two-laptop evaluation, reserved trips, organiser scoring confirmation and teammate publication remain outstanding.")
    r.para("AI assistance was used for implementation, execution of experiments, analysis and report drafting. The repository records the runs. Each teammate must review and reproduce their assigned work and explain their own conclusions before submission.")
    r.start("2.2  /  Dataset understanding","What the raw files contain","Evidence  |  experiments/manifest.csv, inventory.csv, input_identity.json and sources.json")
    r.table([["Partition","Trips","GPS rows","Nodes","Directed arcs","Route entries"]]+[[p,len(z),number(z.observations.sum()),number(z.nodes.sum()),number(z.arcs.sum()),number(z.route.sum())] for p,z in inv.groupby("partition",sort=False)],[.18,.1,.17,.18,.19,.18])
    r.para("Each trip has four headerless whitespace files. A track row stores longitude, latitude and seconds. Node IDs and arc IDs are zero-based row positions. An arc stores its from-node and to-node. A route lists traversed arc IDs in occurrence order. Trip 00000002 has 7,243 observations and 262 route entries. Pairing rows by position would assign incorrect labels.")
    r.para("The audit verified SHA-256 for each of 360 extracted files against its ZIP member. All coordinates and times were finite, indices were in bounds, and timestamps increased within each trip. No GPS rows needed missing-value imputation. Repeated coordinates were retained. Duplicate directed edges retain distinct IDs.")
    r.table([["Partition","Zero length","Duplicate arcs","Reverse exists","Weakly split","Ref. breaks"]]+[[p,str(z.zero_arcs.sum()),number(z.duplicate_arcs.sum()),number(z.reverse_arcs.sum()),f"{(z.weak>1).sum()}/{len(z)}",str(z.breaks.sum())] for p,z in inv.groupby("partition",sort=False)],[.19,.15,.17,.18,.16,.15])
    r.para("A reverse exists when a directed arc has an edge in the opposite direction. Reference breaks count successive route entries whose endpoint IDs do not meet. One such break occurs in student-test trip 00000076. Removing identical-endpoint geometry from spatial searches leaves the original arc numbers intact.")
    q=csv("quality.csv")
    r.para(f"Each trip uses a WGS84 azimuthal equidistant projection centred at its median observed longitude and latitude. This handles different countries and hemispheres. A sampled geodesic cross-check across {number(q.geodesic_checks.sum())} arcs gave a maximum relative length discrepancy of {100*q.max_relative_projection_error.max():.4f}%. Distances in features and routing are metres.")
    r.image("gap",140,"The largest training gap is 153,064 seconds. Each dot is an observed interval or GPS location; the geographic view draws no link across the gap.")
    r.start("2.2  /  Exploration","Road geometry and reference breaks","Evidence  |  experiments/inventory.csv and figures/*_eda.png")
    r.image("train_eda",242,"Training trip 00000002. Grey segments show the network, purple shows reference arcs, orange points show GPS, and crosses mark a reference discontinuity.")
    r.image("student_test_eda",242,"Student-test trip 00000076. Original arc identities matter at crossings and parallel roads. Each plotted reference segment uses its actual endpoints.")
    r.para("A connected-looking drawing can still contain a directed break. The audit checks endpoint identities and both weak and strong graph connectivity. The per-trip inventory also retains isolated nodes, duplicate coordinates, sampling statistics, extents and geodesic lengths.")
    r.start("2.2  /  Preprocessing and supervision","From routes to approximate point labels","Evidence  |  src/alignment.py, src/geometry.py, experiments/alignment.csv, quality.csv and candidate_pilot.csv")
    r.para("The supplied documents describe route sequences. No official observation-to-arc mapping or scoring implementation was supplied. Label preparation projects each observation against usable reference occurrences, ranks them by perpendicular distance plus a directed heading cost, and retains up to 16 occurrences. Repeated traversals remain separate states.")
    r.para("A dynamic program minimises total mismatch. It permits staying or skipping forward in occurrence order. Progress uses cumulative route metres and projection fraction. Transitions penalise differences from observed displacement, permit 15 m of within-segment jitter, and remain inside one connected reference piece. A time gap above 60 s or an otherwise impossible transition starts a fresh block. Block endpoints are free.")
    r.table([["Partition","Aligned within 50 m","Distant","Ambiguous flags","Coverage"]]+[[p,number(z.labelled.sum()),number(z.distant.sum()),number(z.ambiguous.sum()),pct(z.labelled.sum()/z.observations.sum())] for p,z in lab.groupby("partition",sort=False)],[.20,.27,.14,.22,.17])
    r.para("The frozen rule excludes matches farther than 50 m from point scoring. A competing arc within two units of local cost flags ambiguity. Ambiguous matches within 50 m stay in every model's denominator. The 16-occurrence approximation and local heading preference can bias labels. Accuracy against independent point labels is unknown. Exact sequence edit distance uses the supplied route itself.")
    r.head("Network candidates and features")
    pilot=csv("candidate_pilot.csv").groupby(["radius","k"]).mean(numeric_only=True).reset_index()
    r.table([["Initial radius","Maximum candidates","Training recall","Time / pilot trip"]]+[[f"{int(v.radius)} m",int(v.k),pct(v.recall),f"{v.seconds:.3f} s"] for v in pilot.itertuples()],[.21,.29,.23,.27])
    r.para("Five evenly spaced training trips informed the candidate budget. The smallest cap within 0.5 percentage points of the best recall selected 50 m and 12 arcs. An STRtree searches segment interiors. Exact projection ranks ties by original arc ID. Empty queries expand once to 160 m. Remaining misses are unmatched.")
    r.para("Fourteen features describe absolute and relative distance, fraction along arc, length, sine/cosine of heading error, heading validity, displacement, time, speed, turn agreement, two node degrees and expansion. A centred five-point window estimates offline heading within each time block. Displacements under 4 m suppress unreliable window headings. IDs identify rows only.")
    r.start("3.1  /  Model comparison","Five classifiers on the same trips","Evidence  |  experiments/models.csv, folds.json, summary.csv, route_summary.csv and runs/")
    names=["Logistic","GaussianNB","DecisionTree","RandomForest","HistBoost"]
    r.table([["Model","CV F1","Fold SD","Top-1","Dev F1","Continuity","Bin. loss","Score ms"]]+[[m,val(s.loc[m,"cv"].macro_f1),val(s.loc[m,"cv"].cv_fold_sd),val(s.loc[m,"student_test"].accuracy),val(s.loc[m,"student_test"].macro_f1),val(a.loc[m].continuity),val(s.loc[m,"student_test"].binary_log_loss),f"{s.loc[m,'student_test'].inference_s*1000:.1f}"] for m in names],[.18,.1,.1,.105,.105,.13,.13,.15],8)
    r.para("CV F1 is mean held-out trip macro F1. SD uses the five fold means. Other columns use the ten student-test trips. Continuity is the share of eligible transitions admitting a directed path within the movement budget. Score time measures classifier probability inference per trip.")
    r.image("model_comparison",167,"All plotted values are from saved runs. Both loss bars use natural logarithms; binary loss scores relevance rows and conditional loss scores the correct arc among available candidates.")
    r.head("Experiment design")
    r.para(f"Five GroupKFold splits keep whole trips apart. Fitting sampled up to 320 complete observation groups per training trip with seed 352. The final fit used {number(fit['groups'])} groups and {number(fit['rows'])} candidate rows ({number(fit['positives'])} positive). Every model kept the same natural class distribution, including all-negative groups. Every validation and student-test observation was evaluated.")
    r.para("Logistic regression and Gaussian NB fitted their scaler only on each training fold. Features are finite by construction, and supervised calibration and class reweighting were omitted. Boosting internal early stopping was disabled. Selection maximised mean trip-local arc macro F1, breaking ties by lower candidate log loss and then model name. Classifier predictions use the highest candidate score.")
    r.para(f"The additional nearest-segment baseline reached Top-1 {pct(a.loc['Nearest'].accuracy)} and macro F1 {val(a.loc['Nearest'].macro_f1)}. Opposite directions on the same geometry can tie for distance.")
    r.start("3.2  /  Individual experiments","What each classifier learned","Evidence  |  experiments/runs/<model>/configuration.json, train_refit.csv and student_test.csv")
    details={
      "Logistic":("Linear relevance model","Scaled inputs, C=1, 300 iterations, L2-regularised binary logistic loss.","A cheap, interpretable score combined distance and heading. Its linear feature effects gave the lowest arc macro F1 of the five, despite fast scoring."),
      "GaussianNB":("Gaussian naive Bayes","Scaled inputs, variance smoothing 1e-7. Fit class frequencies and per-class Gaussian means and variances by maximum likelihood.","Fast fitting and high recall suit an early comparison. Distance and relative distance are correlated. The observed high binary loss makes its raw probabilities problematic as decoder weights."),
      "DecisionTree":("Single decision tree","Gini impurity reduction, depth 12, at least 20 training rows per leaf.","Threshold splits captured nonlinear distance and motion patterns. Piecewise leaf frequencies yielded a substantial conditional loss. Scoring was the fastest of the five."),
      "RandomForest":("Bagged decision trees","32 trees, depth 14, minimum leaf size 6, square-root feature sampling, two workers. Bootstrap Gini splits average leaf probabilities.","Averaging trees improved held-out F1 over the single tree and achieved the highest candidate precision. Its five-fold fit took longer than the other initial models."),
      "HistBoost":("Histogram gradient boosting","Binary log loss, 70 iterations, learning rate 0.10, 23 leaves, L2=1.0. Early stopping disabled.","Successive trees fit remaining loss gradients. It led on CV F1 and had the lowest initial binary and conditional losses. The score cost exceeded that of the single tree.")}
    for i,m in enumerate(names):
        if i==3:
            r.start("3.2  /  Individual experiments","Ensembles and probability quality","Evidence  |  experiments/summary.csv, models.csv and run configurations")
        v=s.loc[m,"student_test"]; cv=s.loc[m,"cv"]; train=s.loc[m,"train_refit"]
        title,config,obs=details[m];r.head(title);r.para(config+" "+obs)
        r.para(f"Student-test Top-1 {pct(v.accuracy)}, local macro F1 {val(v.macro_f1)}, binary loss {val(v.binary_log_loss)}, conditional observation loss {val(v.conditional_obs_log_loss)}. Final fit {v.fit_s:.3f} s. Full training-trip refit F1 {val(train.macro_f1)} versus held-out CV F1 {val(cv.macro_f1)} (gap {train.macro_f1-cv.macro_f1:+.3f}).",9.8)
        if i==2:r.para("Refit scores include both sampled and unsampled observations from fitted trips. Gaps describe the observed change on held-out trips. They cannot isolate overfitting from differences among trips. None of these fits recorded comparable epoch-wise training and validation losses, so loss-history plots are N/A.")
    r.table([["Student-test rows","Precision","Recall","Binary macro F1","Conditional loss"]]+[[m,val(s.loc[m,"student_test"].precision),val(s.loc[m,"student_test"].recall),val(s.loc[m,"student_test"].candidate_macro_f1),val(s.loc[m,"student_test"].conditional_obs_log_loss)] for m in names],[.25,.16,.14,.23,.22])
    denom=s.loc["Tuned","student_test"]
    r.para(f"The table averages within-trip scores over ten trips. Binary metrics use {number(denom.candidate_rows)} candidate rows from labelled observations, threshold 0.5. Conditional loss covers {pct(denom.candidate_coverage)} of labelled observations by count. The remaining candidate misses count as Top-1 errors. Out-of-fold predictions preserve every observation index.")
    r.start("4  /  Bayesian optimisation","Tuning the validation winner","Evidence  |  experiments/tuning.sqlite3, trials.csv, tuned.csv and artifacts/model.json")
    best=conf["parameters"]
    r.para(f"Optuna's seeded TPE sampler ran {len(trials)} completed trials, starting with five random trials. Each trial refit five disjoint training-trip folds and maximised mean trip-local observation macro F1. Trial {conf['best_trial']} attained {conf['cv_f1']:.5f}. Total recorded trial duration was {pd.to_timedelta(trials.duration).dt.total_seconds().sum():.1f} seconds.")
    r.table([["Parameter","Search space","Best value"]]+[[x,space,f"{best[x]:.5g}"] for x,space in [("max_iter","35-110, integer"),("max_leaf_nodes","9-43, integer"),("learning_rate","0.035-0.22, log-uniform"),("l2_regularization","0.01-12, log-uniform")]],[.31,.42,.27])
    r.para("Iteration and leaf budgets bracketed the initial fit to bound repeated CV cost. Log scales explored proportional changes in shrinkage and regularisation. No evaluator results informed these ranges.")
    r.image("tuning",165,"The line retains the best validation score seen so far. The SQLite study records each trial state, parameter draw, fold score and duration.")
    r.table([["Configuration","CV F1","Dev Top-1","Dev F1","Binary loss","Cond. loss","Score ms"]]+[[m,val(s.loc[m,"cv"].macro_f1),val(s.loc[m,"student_test"].accuracy),val(s.loc[m,"student_test"].macro_f1),val(s.loc[m,"student_test"].binary_log_loss),val(s.loc[m,"student_test"].conditional_obs_log_loss),f"{s.loc[m,'student_test'].inference_s*1000:.1f}"] for m in ["HistBoost","Tuned"]],[.19,.12,.15,.12,.14,.14,.14],8)
    r.para("Tuning yielded small gains on both partitions at a higher scoring cost. The same training folds supported selection and tuning, so the best observed CV score is selection-conditioned. The fitted pipeline, exact schema, runtime versions and SHA-256 are in artifacts/.")
    r.start("5  /  HMM and Viterbi","How a sequence becomes a road path","Evidence  |  src/decoder.py, artifacts/decoder.json, experiments/ablation.csv and profiles.csv")
    r.para("A hidden state is a candidate directed arc and its projected position. Clip positive-class scores to [1e-8, 1-1e-8], then divide by the sum within the observation. These conditional emission weights have no demonstrated probability calibration.")
    r.para("Moving forward on the same arc costs the travelled fraction of its length. Other moves traverse the unused tail of the previous arc, a directed shortest path between endpoints, then the head of the next arc. Backward motion on one arc and opposite-direction arcs require real directed connectivity. Equal-length parallel edges retain original IDs.")
    r.para("For elapsed time dt, routes longer than 40 + 55 dt metres are infeasible. For a feasible distance D, the transition log weight is -(|D - GPS displacement| + max(0, D - 35 dt)) / beta. Directed Dijkstra searches stop at a rounded-up radius, with exact state-pair budgets checked afterwards. A graph-local bounded cache stores up to 2,048 source/radius searches. Chunked queries avoid all-pairs distances.")
    r.para("For each state j, add its log emission to the maximum of the preceding state score plus transition(i,j). Record the winning predecessor, then backtrack at each block end. An empty set, a gap over 60 s or no feasible transition ends the chain. The map draws separate pieces. Unmatched points retain their observation index.")
    r.table([["Dev method","Top-1","Macro F1","Continuity","Edit error","Restarts"]]+[[m,pct(a.loc[m].accuracy),val(a.loc[m].macro_f1),pct(a.loc[m].continuity),val(a.loc[m].edit_error),number(a.loc[m].restarts)] for m in ["Tuned","HMM-8","HMM-16","HMM-32"]],[.23,.15,.16,.16,.15,.15])
    r.para(f"Beta {dconf['beta']} m gave the best development macro F1. Beta 8 m had the lowest edit error. The same ten trips informed this choice, so these decoder results are development estimates. Feasibility counts {number(h.transitions)} adjacent pairs after excluding {int(h.gap_boundaries)} time boundary. All {number(h.observations)} observations have predictions.")
    r.para("Sequence error is exact Levenshtein distance divided by the longer compressed sequence length. Consecutive duplicates collapse within each connected piece; later revisits remain. The compiled RapidFuzz implementation uses compact distance computation. Both methods insert actual graph connectors where feasible. Comparing concatenated pieces cannot fully describe break placement.")
    r.start("5  /  Decoder ablation","Which trips gained or lost accuracy?","Evidence  |  experiments/ablation.csv, route_summary.csv, profiles.csv and http.csv")
    ar=ab.pivot(index="trip",columns="method",values=["accuracy","macro_f1","edit_error"])
    r.table([["Student-test trip","Classifier Top-1","Viterbi Top-1","Change","Viterbi F1","Edit error"]]+[[trip,pct(v["accuracy","Tuned"]),pct(v["accuracy",dec]),f"{100*(v['accuracy',dec]-v['accuracy','Tuned']):+.1f} pp",val(v["macro_f1",dec]),val(v["edit_error",dec])] for trip,v in ar.iterrows()],[.21,.18,.17,.15,.15,.14])
    r.para("Trip 00000077 fell from 68.7% to 57.1% Top-1. A feasible path can stay on a different nearby road than the approximate reference labels. This is a possible explanation; neither the point labels nor the decoded route were independently surveyed. Trip 00000052 gained accuracy. Point-frequency weighting changes the aggregate because long trips contribute more observations.")
    r.table([["Observation-weighted","Top-1","Trip F1 weighted by labels","Feasible / eligible"]]+[[m,pct(a.loc[m].weighted_accuracy),val(a.loc[m].weighted_macro_f1),pct(a.loc[m].weighted_continuity)] for m in ["Tuned",dec]],[.27,.15,.32,.26])
    profile=csv("profiles.csv")
    r.para("Fresh inference profiles include projection, spatial indexing, feature generation, class scoring and directed decoding. Cold classifier inference remains distinct from the warm graph-cache times in the ablation file.")
    r.table([["Profiled training trip","GPS rows","Network arcs","Features","Classifier","Decoder"]]+[[v.trip,number(v.observations),number(v.arcs),f"{v.features_s:.3f} s",f"{v.classifier_s:.3f} s",f"{v.decoder_s:.3f} s"] for v in profile[profile.trip.isin(["00000054","00000013"])].itertuples()],[.26,.15,.18,.14,.14,.13],8)
    r.start("6  /  Folium geographical visualisation","Read the map in layers","Evidence  |  scripts/figures.py, src/maps.py and figures/*_map.png")
    r.image("student_test_map",275,"Offline development view of student-test 00000076. Raw GPS, full road network, reference and predicted route can be toggled independently.")
    r.image("train_map",210,"Training 00000002. The interactive HTML includes a switch for ambiguous reference-derived labels. Routing uses every observation; only displayed GPS markers are thinned.")
    r.para("The geometry layers render without remote tiles. Road directions share one drawing per undirected node pair for display. Directed arc identity remains intact during inference. All displayed positions use latitude then longitude. The private reference layer is confined to offline development maps.")
    r.start("7  /  API and user interface","A request from the trip selector","Evidence  |  figures/ui.png, experiments/http.csv and ui_runs.json")
    r.image("ui",340,"The actual running Streamlit client after a development prediction. The map displays only permitted road geometry, GPS and the decoded route.")
    r.para("Laptop 2 uses requests to GET /health and /trips, then POST /predict with the selected ID and server mode. Laptop 1 loads the fitted model once. Every request reads permitted inputs and computes fresh predictions. Only after predictions are frozen can the private scoring adapter open labels. The response contains route IDs, geometry pieces, diagnostics, aggregate scores and timings.")
    r.table([["Single-host timing","Mean / trip"]]+[[label,f"{http[col].mean():.3f} s"] for label,col in [("Input read","read_s"),("Geometry and features","features_s"),("Classifier","classifier_s"),("Decode + reconstruction","decoder_s"),("Private scoring","scoring_s"),("HTTP round trip","http_s")]]+[["UI click through map ready",f"{np.mean([x['ui_s'] for x in ui]):.3f} s"]],[.72,.28])
    r.para(f"Model startup load was {read('server_health.json')['model_load_s']:.3f} s. HTTP time includes serialisation and transport. Ten development selections reached rendered Folium maps. All timings are local to this MacBook Pro, Apple M1 Pro, ten CPU cores, 16 GB RAM, macOS 15.0.1, Python 3.13.0.",9.5)
    r.start("7  /  Deployment and verification","Receiving-server evidence","Evidence  |  figures/server_terminal.png, experiments/terminal_requests.json, reload.json and tests.log")
    r.image("server_terminal",294,"Dedicated macOS Terminal running Uvicorn. The ten successful HTTP POSTs are from the second local server process. Request IDs link the screenshot to terminal_requests.json.")
    r.para("The real-device configuration binds Uvicorn to 0.0.0.0 on Laptop 1. Laptop 2 uses its LAN IPv4 address. Enable the specific server port on the host firewall. A campus network with client isolation can block device-to-device connections even when both devices have internet access.")
    r.head("Acceptance on the two physical laptops")
    r.para("1. Start Laptop 1 in evaluator mode with a ten-trip input manifest, private reference directory and the agreed organiser mapping. Check /health from Laptop 2.<br/>2. Select each reserved trip. Confirm that the receiving terminal logs the request ID, trip, count, timings and aggregate result.<br/>3. Inspect route pieces and the scoring label. Capture the two physical screens and record the LAN setup.")
    r.para("Development verification includes repeated requests, rejection of traversal and unknown IDs, mode mismatch, separate server processes and a same-port restart. Nine focused tests exercise original IDs, projection order, route occurrence alignment, gaps, directed impossibilities, group folds, scoring denominators, artifact loading, and server-side label isolation.")
    r.head("Scoring contract and outstanding evidence")
    r.para("Development scoring uses approximate route alignment. Evaluator mode can accept an organiser-confirmed two-column observation-index/arc-ID mapping. Without that mapping it returns an explicit unavailable score and the computed route. No reserved evaluator scores or physical LAN success are claimed.")
    r.start("Evidence and viva","How to reproduce the claims","Evidence  |  README.md, HANDOFF.md, experiments/runtime.json and config/research.txt")
    r.para("The supplied MP4 was inspected across its 51.47 seconds. Local Whisper transcription with two decoding settings identified GPS trajectory formation (0-13 s), GPS drift (13-25 s), the road-identification problem (25-33 s) and reconstruction of the likely road sequence from the GPS and network (33-49 s). No extra numerical scoring or deployment rule was identified. The written specifications govern the experiments.")
    r.table([["Question","Evidence to inspect"]]+[list(i) for i in [
        ("Which raw inputs were used?","manifest.csv, input_identity.json, sources.json"),("What was frozen before comparison?","feature_identity.json, config/settings.json, folds.json"),("How was the winner selected?","models.csv, summary.csv, runs/*/cv*.npz"),("Did tuning help?","tuning.sqlite3, trials.csv, tuned.csv"),("What produces route continuity?","src/decoder.py, ablation.csv, route_summary.csv"),("Did a fresh process agree?","http.csv, reload.json, terminal_requests.json"),("Can the demo run offline?","src/maps.py, src/vendor/, scripts/browser_check.py")]], [.45,.55],8.1)
    r.head("Questions for the viva")
    r.para("Why are arc IDs trip-local? Trace one original arc row through projection, candidate scoring and output. Why can a correct arc be unavailable? Show a bounded search miss and how the denominator handles it. Why use grouped validation? Demonstrate one trip fold and its disjoint training manifest.")
    r.para("Why can continuity rise while accuracy falls? Use trip 00000077 and the transition budget. Why is a predicted probability conditional? Show candidate normalisation and distinguish conditional observation loss from binary row loss. Why are two laptop tests still needed? Trace the HTTP request across a real network and identify the separate private scoring read.")
    r.head("Rebuild")
    r.para("Use the pinned research environment and external raw archive. README lists the ordered commands for audit, preparation, training, ablation, checks, map capture and PDF generation. The committed source and outputs require all four handoffs to be merged for a complete public checkout.")
    r.para("The technical team should add its own review notes or corrections and re-run the report generator after the physical acceptance test. No external reference routes, paid location services or cross-team trained models were used.")
    r.finish()
if __name__=="__main__": main()
