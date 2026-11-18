"""PathSnap evaluator interface. Runs on Laptop 2 and calls the FastAPI server on Laptop 1."""
import csv
import io
import math
import os
from datetime import datetime
from time import perf_counter
from urllib.parse import urlparse
import requests
import streamlit as st
from src.maps import route_map,LAYERS

st.set_page_config(page_title="PathSnap · Route reconstruction",layout="wide",initial_sidebar_state="expanded")
st.markdown("""<style>
.block-container{max-width:1400px;padding-top:1.4rem;padding-bottom:2rem}
h1{letter-spacing:-.03em;margin-bottom:0;padding-bottom:0;font-size:2.1rem}
div[data-testid=stMetricValue]{font-size:1.55rem}
div[data-testid=stMetric]{background:rgba(127,127,127,.08);border:1px solid rgba(127,127,127,.25);border-radius:10px;padding:.65rem .9rem}
.ps-chip{display:inline-block;padding:.2rem .65rem;border-radius:999px;font-size:.82rem;font-weight:600;margin-right:.35rem}
.ps-ok{background:#e6f4ea;color:#1a7f37}.ps-bad{background:#fde8e8;color:#b42318}.ps-info{background:#eef2f7;color:#344054}
.ps-sub{opacity:.75;font-size:.95rem;margin-top:.15rem}
</style>""",unsafe_allow_html=True)

DATASET={"development":("Student-test trips","development mode · approximate reference scoring"),
         "evaluator":("Reserved evaluation trips","evaluator mode · organiser scoring")}

def explain(exc,url):
    """Turn a requests failure into a message a demo operator can act on."""
    host=urlparse(url).netloc or url
    if isinstance(exc,requests.ConnectionError):
        port=urlparse(url).port or 80
        return ("Can't reach the API server",f"Nothing answered at **{host}**. On Laptop 1 check that uvicorn is running with "
                f"`--host 0.0.0.0 --port {port}`, that both laptops are on the same network, and that the firewall allows port {port}.")
    if isinstance(exc,requests.Timeout):
        return ("The server took too long","No reply before the timeout. The server may be busy with another trip or the network is slow. Retry once.")
    if isinstance(exc,requests.HTTPError) and exc.response is not None:
        code=exc.response.status_code
        try: detail=exc.response.json().get("detail")
        except ValueError: detail=None
        detail=detail if isinstance(detail,str) else None
        known={404:("Unknown trip","The server doesn't list this trip. Refresh the trip list."),
               409:("Mode mismatch","The server changed mode (development/evaluator). Reload the page."),
               422:("Trip couldn't be processed","The server couldn't read or process this trip's files. Check the data folder on Laptop 1."),
               500:("Server error","The API raised an internal error. Check the uvicorn terminal on Laptop 1.")}
        title,body=known.get(code,(f"Server returned HTTP {code}","Check the uvicorn terminal on Laptop 1."))
        return (f"{title} (HTTP {code})",body+(f" Server said: *{detail}*" if detail else ""))
    return ("Unexpected response","The server replied with something that isn't valid PathSnap JSON. Check the server URL.")

@st.cache_data(ttl=10,show_spinner=False)
def connect(url):
    tic=perf_counter()
    health=requests.get(url+"/health",timeout=5); health.raise_for_status()
    trips=requests.get(url+"/trips",timeout=5); trips.raise_for_status()
    return health.json(),trips.json(),perf_counter()-tic

def length_km(path):
    total=0.0
    for (a,b),(c,d) in zip(path,path[1:]):
        p,q=math.radians(a),math.radians(c)
        h=math.sin((q-p)/2)**2+math.cos(p)*math.cos(q)*math.sin(math.radians(d-b)/2)**2
        total+=2*6371.0088*math.asin(math.sqrt(h))
    return total

# ---------- Sidebar: connection ----------
with st.sidebar:
    st.subheader("Server connection")
    url=st.text_input("API server URL",os.getenv("PATHSNAP_SERVER","http://127.0.0.1:18521"),
                      help="Laptop 1's LAN address, e.g. http://192.168.1.23:18521. 127.0.0.1 only works when the API runs on this laptop.").strip().rstrip("/")
    if st.button("Test connection",width="stretch"): connect.clear()
    status_box=st.container()

try:
    health,available,ping=connect(url)
    mode=available["mode"]
except (requests.RequestException,ValueError,KeyError) as exc:
    with status_box: st.markdown('<span class="ps-chip ps-bad">● Not connected</span>',unsafe_allow_html=True)
    st.title("PathSnap")
    title,body=explain(exc,url)
    st.error(f"**{title}.** {body}")
    st.stop()

dataset,dataset_note=DATASET.get(mode,(mode,""))
with status_box:
    st.markdown(f'<span class="ps-chip ps-ok">● Connected</span><span class="ps-chip ps-info">{ping*1000:.0f} ms</span>',unsafe_allow_html=True)
    st.caption(f"Server **{urlparse(url).netloc}** · mode **{mode}** · status **{health.get('status','?')}** · model loaded in {health.get('model_load_s',0):.2f} s")

# ---------- Header and selection ----------
st.title("PathSnap")
st.markdown('<div class="ps-sub">Reconstructs the road route a vehicle drove from its noisy GPS trail.</div>',unsafe_allow_html=True)
st.write("")
trips=available.get("trips",[])
if not trips:
    st.warning(f"**No trips available.** The server is up in {mode} mode but lists no trips. Check the data folder and manifest on Laptop 1.")
    st.stop()

with st.container(border=True):
    c1,c2,c3=st.columns([2,2,1.2],vertical_alignment="bottom")
    c1.markdown(f"**Dataset**  \n{dataset} · {len(trips)} trips  \n<span style='opacity:.7;font-size:.85rem'>{dataset_note}</span>",unsafe_allow_html=True)
    trip=c2.selectbox("Trip",trips,help="Each trip is one vehicle journey with its own road network.")
    run=c3.button("Run prediction",type="primary",width="stretch")

if run:
    st.session_state.pop("result",None)
    with st.status(f"Running prediction for trip {trip}…",expanded=True) as status:
        st.write(f"Sending trip **{trip}** to **{urlparse(url).netloc}**…")
        tic=perf_counter()
        try:
            resp=requests.post(url+"/predict",json={"trip_id":trip,"mode":mode},timeout=180)
            resp.raise_for_status()
            result=resp.json()
            elapsed=perf_counter()-tic
            st.write(f"Server matched {result['diagnostics']['observations']:,} GPS points in {result['timing']['inference_s']:.2f} s "
                     f"(round trip {elapsed:.2f} s). Drawing map…")
            st.session_state.result=(url,result,elapsed)
            st.session_state.setdefault("history",[]).append({"time":datetime.now().strftime("%H:%M:%S"),"trip_id":result["trip_id"],
                "request_id":result["request_id"],"server":urlparse(url).netloc,"points":result["diagnostics"]["observations"],
                "accuracy":result["metrics"].get("accuracy"),"pieces":result["diagnostics"]["pieces"],"round_trip_s":round(elapsed,3)})
            status.update(label=f"Trip {trip} done · request {result['request_id'][:8]}",state="complete",expanded=False)
        except (requests.RequestException,ValueError,KeyError) as exc:
            title,body=explain(exc,url)
            status.update(label=f"Trip {trip} failed",state="error",expanded=True)
            st.error(f"**{title}.** {body}")

if "result" not in st.session_state:
    st.info("Pick a trip and press **Run prediction**. The map and scores appear here.")
    st.stop()

server,r,elapsed=st.session_state.result
if (server,r["mode"])!=(url,mode):
    st.info("The server or mode changed since the last run. Press **Run prediction** again.")
    st.stop()
if r["trip_id"]!=trip:
    st.info(f"Trip **{trip}** selected. Press **Run prediction** to load trip {trip}.")
    st.stop()

# ---------- Results ----------
d,m,t=r["diagnostics"],r["metrics"],r["timing"]
pieces=r["geometry"]
km=sum(length_km(p) for p in pieces)
st.markdown(f"#### Trip {r['trip_id']} · {dataset}")
st.caption(f"Request ID `{r['request_id']}` — match this against the uvicorn log on Laptop 1.")
k=st.columns(6)
k[0].metric("GPS points",f"{d['observations']:,}",help="Raw GPS observations in this trip.")
k[1].metric("Matched",f"{d['matched']/max(d['observations'],1):.1%}",help=f"{d['matched']:,} points got a road; {d['unmatched']:,} had no nearby road.")
acc=m.get("accuracy")
k[2].metric("Top-1 accuracy",f"{acc:.1%}" if acc is not None else "n/a",help="Share of labelled points whose predicted road equals the reference road. "
            "In development mode the reference is an approximate alignment of the true route.")
k[3].metric("Route continuity",f"{d['continuity']:.1%}",help="Share of consecutive point pairs that are connected by a drivable path.")
k[4].metric("Route pieces",f"{d['pieces']}",help=f"Connected stretches of route, {km:.1f} km in total. Pieces split at GPS gaps or where no drivable link was found.")
k[5].metric("Server time",f"{t['inference_s']:.2f} s",help=f"Model and decoder time on Laptop 1. Round trip including network: {elapsed:.2f} s.")
labelled=m.get("labelled")
st.caption(f"Scoring: {m.get('status','unavailable')}"+(f" · Labelled {labelled:,} of {d['observations']:,}" if labelled is not None else "")
           +(f" · Macro F1 {m['macro_f1']:.3f}" if m.get("macro_f1") is not None else "")
           +(f" · Edit error {m['edit_error']:.3f}" if m.get("edit_error") is not None else "")
           +(f" · {m['reason']}" if m.get("reason") else ""))

# ---------- Map ----------
layers=st.pills("Map layers · click to show or hide",list(LAYERS),selection_mode="multi",default=list(LAYERS),key="layers",
                help="Click to show or hide a layer. The map keeps the same view of the trip.")
if not layers: st.caption("All layers are hidden. Pick at least one above.")
view=route_map(r["gps"],r["network"],pieces,layers=layers or [])
st.iframe(view.get_root().render(),height=640,alt=f"Map of trip {r['trip_id']}: road network, GPS observations and reconstructed route")
st.caption("**How to read the map:** grey lines are every road in the trip's map; orange dots are the raw GPS points; the green line is the route "
           "PathSnap reconstructed. Pink rings mark where one route piece ends and the next begins. Hover a marker for details.")

# ---------- Details ----------
tab1,tab2,tab3=st.tabs(["Route pieces","Request and timing","Run history"])
with tab1:
    st.dataframe([{"Piece":i+1,"Roads (arcs)":len(part),"Length (km)":round(length_km(path),2)}
                  for i,(part,path) in enumerate(zip(r["route_parts"],pieces))],hide_index=True,width="stretch")
    st.caption(f"{d['gap_boundaries']} GPS gaps over 60 s and {d['restarts']} decoder restarts split the route.")
with tab2:
    st.json({"request_id":r["request_id"],"server":server,"trip_id":r["trip_id"],"mode":r["mode"],"round_trip_s":round(elapsed,3),
             "server_seconds":{k:round(v,4) for k,v in t.items()},"diagnostics":d,"metrics":m},expanded=False)
with tab3:
    history=st.session_state.get("history",[])
    st.dataframe(history,hide_index=True,width="stretch")
    buf=io.StringIO()
    if history:
        w=csv.DictWriter(buf,fieldnames=list(history[0])); w.writeheader(); w.writerows(history)
    st.download_button("Download run history (CSV)",buf.getvalue(),file_name="ui_history.csv",mime="text/csv",disabled=not history)
    st.caption("Every prediction made from this browser tab. Match it against Laptop 1's log with `python -m scripts.match_requests ui_history.csv server.log`.")
