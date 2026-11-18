"""Additional local HTTP, browser failure/recovery and privacy acceptance checks."""
import json,asyncio,argparse
from pathlib import Path
import requests,numpy as np
from playwright.sync_api import sync_playwright
from src.data import DATA,read_inputs
parser=argparse.ArgumentParser()
parser.add_argument('--lan-url',required=True)
args=parser.parse_args()
out=Path('experiments/acceptance');trips=requests.get('http://127.0.0.1:18521/trips').json()['trips']; checks=[]
for trip in trips:
 r=requests.post('http://127.0.0.1:18521/predict',json={'trip_id':trip,'mode':'development'}).json(); d=read_inputs(DATA/'student_test',trip)
 for part in r['route_parts']:
  a=d.arcs[part];assert (a[:-1,1]==a[1:,0]).all()
 assert len(r['gps'])==len(d.track)
 for bad in ['reference_arc_ids','truth','correctness','private_path','/Users/']:assert bad not in json.dumps(r)
 checks.append(dict(trip=trip,connected_parts=True,private_labels_absent=True,observation_count=len(d.track)))
assert requests.post('http://127.0.0.1:18521/predict',json={'trip_id':trips[0],'mode':'development','extra':True}).status_code==422
lan=requests.get(args.lan_url.rstrip('/')+'/health',timeout=5).json();assert lan['status']=='ready'
with sync_playwright() as p:
 b=p.chromium.launch(channel='chrome',headless=True);page=b.new_page();page.goto('http://127.0.0.1:18522');page.get_by_role('button',name='Run prediction').wait_for()
 page.get_by_role('button',name='Expand sidebar').click()
 box=page.get_by_role('textbox',name='API server URL');box.fill('http://127.0.0.1:1');box.press('Enter')
 page.get_by_text('API unavailable.',exact=False).wait_for();page.screenshot(path=str(out/'server_unavailable.png'))
 box.fill('http://127.0.0.1:18521');box.press('Enter');page.get_by_role('button',name='Run prediction').wait_for();b.close()
# Exercise full inference before missing private scoring, without any .route read.
from src import api
async def missing():
 async with api.lifespan(api.app):
  api.app.state.private=None
  r=api.run(api.Request(mode='development',trip_id=trips[0]));assert r['metrics']['status']=='unavailable' and len(r['predicted_arc_ids'])==2503
asyncio.run(missing())
(out/'extra_checks.json').write_text(json.dumps(dict(trips=checks,extra_field_rejected=True,server_unavailable_ui=True,reconnection=True,missing_private_prediction_success=True,same_mac_lan_health=lan,lan_url=args.lan_url,second_physical_device=False),indent=2))
