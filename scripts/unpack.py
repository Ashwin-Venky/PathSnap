"""Restore the two raw partitions. Reject unexpected file layouts and changed originals."""
import argparse
from pathlib import Path,PurePosixPath
import re
from zipfile import ZipFile
from src.data import digest
import hashlib

def restore(archive,dest):
    members=[]
    with ZipFile(archive) as z:
        for info in z.infolist():
            if info.is_dir(): continue
            parts=PurePosixPath(info.filename).parts
            if parts[:1]==("Student folder",): parts=parts[1:]
            if len(parts)!=3 or parts[0] not in {"train","student_test"} or not re.fullmatch("[0-9]{8}",parts[1]) or parts[2] not in [parts[1]+"."+x for x in ("track","nodes","arcs","route")]:
                raise ValueError("Unexpected archive member")
            out=dest.joinpath(*parts)
            if not out.resolve().is_relative_to(dest.resolve()): raise ValueError("Unsafe destination")
            members.append((info,out))
        if len({p for _,p in members})!=360 or len(members)!=360: raise ValueError("Expected 360 distinct files")
        for info,out in members:
            if out.exists() and digest(out)!=hashlib.sha256(z.read(info)).hexdigest():
                raise ValueError(f"Existing file differs: {out.name}")
        for info,out in members:
            if out.exists():continue
            out.parent.mkdir(parents=True,exist_ok=True)
            with out.open("xb") as f: f.write(z.read(info))
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("archive",type=Path)
    p.add_argument("destination",type=Path)
    a=p.parse_args()
    restore(a.archive,a.destination)
