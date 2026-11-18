import asyncio
import os
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch
from pathlib import Path
import numpy as np
from src import api
from src.data import Inputs
from src.private_score import score

class InterfaceChecks(unittest.TestCase):
    def test_predict_finishes_before_private_read(self):
        events=[]
        data=Inputs(np.array([[10.,50.,1]]),np.array([[10.,50.],[10.01,50.]]),np.array([[0,1]]))
        def infer(inputs,model,beta,projection_tolerance_m):
            self.assertIs(inputs,data)
            self.assertEqual(projection_tolerance_m,25)
            events.append('prediction')
            return np.array([0]),[[0]],{'observations':1},{'inference_s':.1}
        def private(mode,trip,root,inputs,ids,parts):
            self.assertFalse(ids.flags.writeable)
            self.assertEqual(parts,((0,),))
            events.append('scoring')
            return {'status':'unavailable'}
        async def run():
            with patch.dict(os.environ,{'PATHSNAP_MODE':'development'}):
                async with api.lifespan(api.app):
                    with patch.object(api,'read_inputs',return_value=data),patch.object(api,'predict',side_effect=infer),patch.object(api,'score',side_effect=private):
                        out=api.Result(**api.run(api.Request(mode='development',trip_id=api.app.state.trips[0])))
                        self.assertEqual(out.gps,[[50.,10.]])
                        self.assertEqual(out.metrics,{'status':'unavailable'})
        asyncio.run(run())
        self.assertEqual(events,['prediction','scoring'])
    def test_missing_evaluator_and_official_mapping(self):
        async def absent():
            with patch.dict(os.environ,{'PATHSNAP_MODE':'evaluator','PATHSNAP_EVAL_INPUT':'','PATHSNAP_EVAL_MANIFEST':'/nonexistent'}):
                async with api.lifespan(api.app): self.assertEqual(api.health()['status'],'trips unavailable')
        asyncio.run(absent())
        data=Inputs(np.zeros((1,3)),np.zeros((2,2)),np.array([[0,1]]))
        with TemporaryDirectory() as directory:
            trip='00000111'
            self.assertEqual(score('evaluator',trip,directory,data,np.array([0]),[])['status'],'unavailable')
            p=Path(directory)/trip
            p.mkdir()
            (p/(trip+'.points')).write_text('0 0\n')
            self.assertEqual(score('evaluator',trip,directory,data,np.array([0]),[])['accuracy'],1.)
            (p/(trip+'.points')).unlink()
            (p/(trip+'.route')).write_text('0\n')
            route_data=Inputs(np.array([[10.,50.,1.]]),np.array([[10.,50.],[10.001,50.]]),np.array([[0,1]]))
            approximate=score('evaluator',trip,directory,route_data,np.array([0]),[[0]])
            self.assertEqual(approximate['status'],'approximate reference-derived accuracy')
            self.assertEqual(approximate['coverage'],1.)
