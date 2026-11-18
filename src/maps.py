"""Offline Folium layers; all coordinates received here are latitude, longitude."""
from pathlib import Path
import folium
from branca.element import Element

PALETTE={"Road network":"#c4c5bf","Predicted route":"#007563","Raw GPS":"#d0652a","Reference route":"#74499b",
         "Ambiguous labels":"#bb3156","Start":"#1a7f37","End":"#1f2328","Route breaks":"#c2185b"}
LAYERS=("Road network","Raw GPS","Predicted route","Start and end","Route breaks")
VENDOR=Path(__file__).parent/"vendor"

def _points_layer(m,label,pts,radius=2,opacity=.75):
    layer=folium.FeatureGroup(name=label).add_to(m)
    # Evenly thin markers only for drawing. Preserve both endpoints.
    stride=max(1,(len(pts)+699)//700)
    shown=pts[::stride]
    if len(pts)>1 and (len(pts)-1)%stride: shown=shown+[pts[-1]]
    coords=[[p[1],p[0]] for p in shown]
    if coords:
        folium.GeoJson({"type":"Feature","properties":{"style":{}},"geometry":{"type":"MultiPoint","coordinates":coords}},
             marker=folium.CircleMarker(radius=radius,color=PALETTE[label],weight=0,fill=True,fill_opacity=opacity)).add_to(layer)

def route_map(gps,network,geometry,reference=None,ambiguous=None,layers=None):
    """Draw network, GPS and route. `layers` limits which of LAYERS (plus reference layers) are drawn."""
    show=set(LAYERS) if layers is None else set(layers)
    if reference is not None and layers is None: show|={"Reference route","Ambiguous labels"}
    m=folium.Map(location=gps[0],tiles=None,prefer_canvas=True,zoom_start=14,control_scale=True,zoom_animation=False,fade_animation=False,zoom_snap=.25)
    m.default_js=[]
    m.default_css=[]
    head=m.get_root().header
    head.add_child(Element("<script>"+(VENDOR/"leaflet.js").read_text()+"</script>"))
    css=(VENDOR/"leaflet.css").read_text()
    head.add_child(Element("<style>"+css+".leaflet-container{background:#f6f5ef}.leaflet-control-layers-toggle{background-image:none!important}"
        ".leaflet-control-layers-toggle:after{content:'☰';display:grid;place-items:center;height:100%;font-size:20px}</style>"))
    # Canvas draws in insertion order: network, then GPS, then the route on top so it stays visible.
    if network and "Road network" in show:
        layer=folium.FeatureGroup(name="Road network").add_to(m)
        folium.PolyLine(network,color=PALETTE["Road network"],weight=1.3,opacity=.9).add_to(layer)
    if reference and "Reference route" in show:
        layer=folium.FeatureGroup(name="Reference route").add_to(m)
        folium.PolyLine(reference,color=PALETTE["Reference route"],weight=6,opacity=.6).add_to(layer)
    if gps and "Raw GPS" in show: _points_layer(m,"Raw GPS",gps)
    if ambiguous is not None and "Ambiguous labels" in show: _points_layer(m,"Ambiguous labels",ambiguous,radius=3)
    if geometry and "Predicted route" in show:
        layer=folium.FeatureGroup(name="Predicted route").add_to(m)
        folium.PolyLine(geometry,color="white",weight=7,opacity=.9).add_to(layer)  # halo for contrast
        folium.PolyLine(geometry,color=PALETTE["Predicted route"],weight=4,opacity=1).add_to(layer)
    if geometry and "Route breaks" in show and len(geometry)>1:
        layer=folium.FeatureGroup(name="Route breaks").add_to(m)
        for i in range(len(geometry)-1):
            for p,where in ((geometry[i][-1],"ends"),(geometry[i+1][0],"starts")):
                folium.CircleMarker(p,radius=5,color=PALETTE["Route breaks"],weight=2,fill=True,fill_color="white",fill_opacity=1,
                    tooltip=f"Break: piece {i+1+(where=='starts')} {where} here").add_to(layer)
    if gps and "Start and end" in show:
        layer=folium.FeatureGroup(name="Start and end").add_to(m)
        for p,label in ((gps[0],"Start"),(gps[-1],"End")):
            folium.CircleMarker(p,radius=8,color="white",weight=3,fill=True,fill_color=PALETTE[label],fill_opacity=1,tooltip=label).add_to(layer)
    legend=[("Road network","line"),("Raw GPS","dot"),("Predicted route","line"),("Start","dot"),("End","dot"),("Route breaks","ring")]
    if reference is not None: legend+=[("Reference route","line"),("Ambiguous labels","dot")]
    shown={"Start":"Start and end","End":"Start and end"}
    def swatch(n,kind):
        c=PALETTE[n]
        if kind=="line": return f'<span style="display:inline-block;width:16px;height:4px;border-radius:2px;background:{c};vertical-align:middle"></span>'
        if kind=="ring": return f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;border:2px solid {c};vertical-align:middle"></span>'
        return f'<span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:{c};vertical-align:middle"></span>'
    items=" &nbsp;&nbsp;".join(f"{swatch(n,k)} {n}" for n,k in legend if shown.get(n,n) in show)
    if items:
        m.get_root().html.add_child(Element('<div style="position:fixed;bottom:10px;left:50%;transform:translateX(-50%);z-index:800;background:rgba(255,255,255,.94);'
            'padding:6px 12px;font:12px -apple-system,Segoe UI,Arial;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.18);white-space:nowrap">'+items+'</div>'))
    m.fit_bounds([[min(x[0] for x in gps),min(x[1] for x in gps)],[max(x[0] for x in gps),max(x[1] for x in gps)]],padding=(24,24))
    folium.LayerControl(collapsed=True).add_to(m)
    return m
