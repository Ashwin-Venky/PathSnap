"""Capture directly used package versions and reproduce laptop-specific requirements."""
from importlib.metadata import version
import platform
from src.data import ROOT,RESULT,save_json
groups={"server":["numpy","scipy","scikit-learn","joblib","shapely","pyproj","rapidfuzz","fastapi","uvicorn","pydantic"],
        "ui":["streamlit","requests","folium","branca"],
        "research":["optuna","pandas","matplotlib","reportlab","Pillow","playwright","threadpoolctl"]}
for group,pkgs in groups.items():
    text="".join(f"{p}=={version(p)}\n" for p in pkgs)
    if group=="research": text="-r server.txt\n-r ui.txt\n"+text
    (ROOT/"config"/(group+".txt")).write_text(text)
save_json(RESULT/"runtime.json",dict(python=platform.python_version(),os=platform.platform(),machine=platform.machine(),
         packages={p:version(p) for group in groups.values() for p in group}))
