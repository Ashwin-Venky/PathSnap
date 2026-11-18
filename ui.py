import os
from time import perf_counter
import requests
import streamlit as st
from src.maps import route_map

st.set_page_config(page_title="PathSnap · Route reconstruction",layout="wide",initial_sidebar_state="collapsed")
st.markdown("<style>.block-container{max-width:1220px;padding-top:2.7rem}h1{letter-spacing:-.04em}div[data-testid=stMetricValue]{font-size:1.6rem}</style>",unsafe_allow_html=True)
st.title("PathSnap")
with st.sidebar:
    st.header("Connection")
    url=st.text_input("API server URL",os.getenv("PATHSNAP_SERVER","http://127.0.0.1:18521")).rstrip("/")
try:
    health=requests.get(url+"/health",timeout=5)
    health.raise_for_status()
    available=requests.get(url+"/trips",timeout=5)
    available.raise_for_status()
    available=available.json()
except (requests.RequestException,ValueError):
    st.error("API unavailable. Check the server URL in Connection and verify that both laptops can reach the server.")
    st.stop()
mode=available["mode"]
st.caption(("Development · Student-test trips" if mode=="development" else "Final evaluation · Reserved trips")+"   /   API connected")
if not available["trips"]:
    st.info("Trip inputs are unavailable for this server mode.")
    st.stop()
left,right=st.columns([4,1],vertical_alignment="bottom")
trip=left.selectbox("Trip",available["trips"])
if right.button("Run prediction",type="primary",width="stretch"):
    st.session_state.pop("result",None)
    try:
        with st.spinner("Reconstructing the directed route…"):
            tic=perf_counter()
            resp=requests.post(url+"/predict",json={"trip_id":trip,"mode":mode},timeout=180)
            resp.raise_for_status()
            st.session_state.result=(url,resp.json(),perf_counter()-tic)
    except (requests.RequestException,ValueError):
        st.error("Prediction failed. Check the selected trip, server logs and network connection, then retry.")
if "result" in st.session_state:
    server,r,elapsed=st.session_state.result
    if (server,r["mode"],r["trip_id"])==(url,mode,trip):
        d,m=r["diagnostics"],r["metrics"]
        cols=st.columns(4)
        cols[0].metric("Observations",f"{d['observations']:,}")
        cols[1].metric("Top-1 accuracy",f"{m['accuracy']:.1%}" if m.get("accuracy") is not None else "Unavailable")
        cols[2].metric("Route continuity",f"{d['continuity']:.1%}")
        cols[3].metric("Inference",f"{r['timing']['inference_s']:.3f} s")
        st.caption(f"Scoring: {m['status']} · Labelled {m.get('labelled','—')} of {d['observations']} · {d['pieces']} route pieces")
        view=route_map(r["gps"],r["network"],r["geometry"])
        st.iframe(view.get_root().render(),height=440,alt="Road network, GPS observations and reconstructed route")
        with st.expander("Request details"):
            st.write({"request_id":r["request_id"],"round_trip_s":round(elapsed,3),"diagnostics":d,"metrics":m,"server_seconds":r["timing"]})
