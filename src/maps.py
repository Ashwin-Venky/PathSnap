"""Offline Folium layers; all coordinates received here are latitude, longitude."""
from pathlib import Path
import json
import folium
from branca.element import Element

PALETTE={"Road network":"#b7b8b3","Predicted route":"#007563","Raw GPS":"#a14316","Reference route":"#74499b","Ambiguous labels":"#bb3156"}
VENDOR=Path(__file__).parent/"vendor"

def route_map(gps,network,geometry,reference=None,ambiguous=None):
    m=folium.Map(location=gps[0],tiles=None,prefer_canvas=True,zoom_start=14,control_scale=True,zoom_animation=False,fade_animation=False,zoom_snap=.25)
    m.default_js=[]
    m.default_css=[]
    head=m.get_root().header
    head.add_child(Element("<script>"+(VENDOR/"leaflet.js").read_text()+"</script>"))
    css=(VENDOR/"leaflet.css").read_text()
    head.add_child(Element("<style>"+css+".leaflet-container{background:#f4f2e9}.leaflet-control-layers-toggle{background-image:none!important}.leaflet-control-layers-toggle:after{content:'☰';display:grid;place-items:center;height:100%;font-size:20px}</style>"))
    for label,lines,width in [("Road network",network,1.4),("Reference route",reference,6),("Predicted route",geometry,3)]:
        if lines:
            layer=folium.FeatureGroup(name=label).add_to(m)
            folium.PolyLine(lines,color=PALETTE[label],weight=width,opacity=.85).add_to(layer)
    for label,pts in [("Raw GPS",gps),("Ambiguous labels",ambiguous)]:
        if pts is None: continue
        layer=folium.FeatureGroup(name=label,show=label=="Raw GPS").add_to(m)
        # Evenly thin markers only for drawing. Preserve both endpoints.
        stride=max(1,(len(pts)+699)//700)
        shown=pts[::stride]
        if len(pts)>1 and (len(pts)-1)%stride: shown=shown+[pts[-1]]
        coords=[[p[1],p[0]] for p in shown]
        if coords:
            folium.GeoJson({"type":"Feature","properties":{"style":{}},"geometry":{"type":"MultiPoint","coordinates":coords}},
                 marker=folium.CircleMarker(radius=2,color=PALETTE[label],weight=0,fill=True,fill_opacity=.8)).add_to(layer)
    names=["Road network","Raw GPS","Predicted route"]+(["Reference route","Ambiguous labels"] if reference is not None else [])
    legend=" &nbsp; ".join(f'<span style="color:{PALETTE[n]}">●</span> {n}' for n in names)
    m.get_root().html.add_child(Element('<div style="position:fixed;top:12px;left:58px;z-index:800;background:white;padding:6px 10px;font:12px Arial;border-radius:4px">'+legend+'</div>'))
    m.fit_bounds([[min(x[0] for x in gps),min(x[1] for x in gps)],[max(x[0] for x in gps),max(x[1] for x in gps)]],padding=(18,18))
    folium.LayerControl(collapsed=True).add_to(m)
    return m
