"""Build the approved same-team review bundle and stage only one owner's exact paths."""
import argparse
import json
import subprocess
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/"config/handoff.json"

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=["list","bundle","stage","check"])
    parser.add_argument("owner",nargs="?")
    args=parser.parse_args()
    owners=json.loads(MANIFEST.read_text())
    all_paths=[p for files in owners.values() for p in files]
    if len(set(all_paths))!=len(all_paths): raise ValueError("A handoff path has more than one owner")
    for p in all_paths:
        path=ROOT/p
        if not path.resolve().is_relative_to(ROOT) or p.startswith((".git/",".cache/",".private/")):
            raise ValueError("Unexpected path")
    if args.action=="bundle":
        target=ROOT/".cache/team-review.zip"
        target.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(target,"w",zipfile.ZIP_DEFLATED) as archive:
            for p in all_paths: archive.write(ROOT/p,p)
        print(target)
        return
    if args.owner not in owners: parser.error("Supply the exact GitHub login from config/handoff.json")
    paths=owners[args.owner]
    if args.action=="list":
        print("\n".join(paths))
        return
    if args.action=="stage": subprocess.run(["git","add","--",*paths],cwd=ROOT,check=True)
    staged=subprocess.check_output(["git","diff","--cached","--name-only","-z"],cwd=ROOT).decode().split("\0")
    unexpected=set(filter(None,staged))-set(paths)
    if unexpected: raise SystemExit("Other paths are staged: "+", ".join(sorted(unexpected)))
    subprocess.run(["git","diff","--cached","--check"],cwd=ROOT,check=True)
    print(f"Checked {sum(bool(p) for p in staged)} staged paths for {args.owner}.")
if __name__=="__main__": main()
