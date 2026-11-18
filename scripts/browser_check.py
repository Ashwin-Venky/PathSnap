"""Drive the running Streamlit UI. Use real browser screenshots and request IDs."""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter
from playwright.sync_api import sync_playwright
from src.data import ROOT,RESULT,save_json

def capture(url):
    trips=json.loads((ROOT/"config/development.json").read_text())["trips"]
    rows=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(channel=os.environ.get("PATHSNAP_BROWSER","chrome"),headless=True)
        page=browser.new_page(viewport={"width":1300,"height":900},device_scale_factor=1)
        page.goto(url)
        page.get_by_role("button",name="Run prediction").wait_for(timeout=45000)
        for i,trip in enumerate(trips):
            tic=perf_counter()
            if i:
                combo=page.get_by_role("combobox")
                combo.click()
                page.get_by_role("option",name=trip,exact=True).click()
                page.get_by_text("Scoring: approximate reference alignment",exact=False).wait_for(state="hidden")
            page.get_by_role("button",name="Run prediction").click()
            label=page.get_by_text("Scoring: approximate reference alignment",exact=False)
            label.wait_for(timeout=90000)
            frame=page.frame_locator("iframe").last
            frame.locator(".leaflet-overlay-pane canvas").wait_for(timeout=45000)
            frame.locator(".leaflet-control-layers").wait_for()
            page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
            if i==0: page.screenshot(path=str(RESULT/"ui.png"))
            rows.append(dict(trip=trip,ui_s=perf_counter()-tic,caption=label.inner_text(),map_rendered=True))
            save_json(RESULT/"ui_runs.json",rows)
        for name in ["train","student_test"]:
            path=ROOT/"maps"/(name+".html")
            if path.exists():
                page.set_viewport_size({"width":1080,"height":650})
                page.goto(path.as_uri())
                page.locator(".leaflet-overlay-pane canvas").wait_for()
                page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
                page.screenshot(path=str(RESULT/(name+"_map.png")))
        browser.close()
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--url",default="http://127.0.0.1:18522")
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    if args.output:
        RESULT=args.output
        RESULT.mkdir(parents=True,exist_ok=True)
    capture(args.url)
