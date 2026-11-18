import json
import unittest
import numpy as np
from src.data import RESULT,ROOT
from src.metrics import classification,point_metrics,select
from src.predict import load_model

class ModelChecks(unittest.TestCase):
    def test_whole_trip_folds(self):
        folds=json.loads((RESULT/'folds.json').read_text())
        holdouts=[]
        for f in folds:
            self.assertFalse(set(f['train'])&set(f['validation']))
            self.assertEqual(len(f['train'])+len(f['validation']),80)
            holdouts.extend(f['validation'])
        self.assertEqual(len(holdouts),len(set(holdouts)))
        self.assertEqual(len(holdouts),80)
    def test_misses_remain_in_denominator(self):
        ids=np.array([[31,42],[11,-1],[-1,-1]])
        p=np.array([[.2,.7],[.1,0],[0,0]])
        m=classification(np.array([42,50,12]),ids,p)
        self.assertEqual(m['labelled'],3)
        self.assertAlmostEqual(m['accuracy'],1/3)
        self.assertAlmostEqual(m['miss_rate'],2/3)
        self.assertAlmostEqual(m['conditional_obs_log_loss'],-np.log(.7/.9))
    def test_fresh_estimator_schema(self):
        self.assertEqual(load_model().n_features_in_,len(json.loads((ROOT/'artifacts/model.json').read_text())['features']))
