import unittest
import numpy as np
from src.data import Inputs
from src.geometry import Geometry,Candidates
from src.decoder import decode,RoadPaths,reconstruct
from src.metrics import edit_error

class DecoderChecks(unittest.TestCase):
    def test_directed_bridge_duplicate_ids_and_breaks(self):
        nodes=np.array([[0.,0.],[.0001,0.],[.0002,0.],[.0003,0.],[.001,0.],[.002,0.]])
        arcs=np.array([[0,1],[1,2],[1,2],[2,3],[4,5]])
        track=np.array([[0.,0.,0],[.0003,0.,1],[.001,0.,2],[.001,0.,100],[.003,0.,101]])
        g=Geometry(Inputs(track,nodes,arcs))
        r=RoadPaths(g)
        c=Candidates(np.array([[0],[3],[4],[4],[-1]]),np.full((5,1),.5),np.zeros((5,1,1)))
        states,cut=decode(g,c,np.ones((5,1)),r)
        ids,parts,d=reconstruct(g,c,states,r,cut)
        self.assertEqual(parts,[[0,1,3],[4],[4]])
        self.assertEqual((d['gap_boundaries'],d['unmatched'],d['feasible']),(1,1,1))
        self.assertIsNone(r.bridge(0,0,.8,.2,500))
        self.assertIsNone(r.bridge(0,3,.2,.8,1))
        self.assertEqual(edit_error([1,1,2,1],[[1,2,1]]),0.)
