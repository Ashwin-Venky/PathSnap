"""Record supplied source fingerprints without redistributing organiser materials."""
from pathlib import Path
from zipfile import ZipFile
import argparse,hashlib
from src.data import RESULT,digest,save_json

def record(raw,guides,video):
    with ZipFile(guides) as z: docs={i.filename:hashlib.sha256(z.read(i)).hexdigest() for i in z.infolist() if i.filename.endswith(".pdf")}
    save_json(RESULT/"sources.json",dict(data_archive=digest(raw),guidance_archive=digest(guides),documents=docs,video_sha256=digest(video),
             video_duration_s=51.47,caption_streams=0,faculty_email="waived by Ashwin",reserved_trips="unavailable"))
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("data",type=Path)
    p.add_argument("guides",type=Path)
    p.add_argument("video",type=Path)
    a=p.parse_args()
    record(a.data,a.guides,a.video)
