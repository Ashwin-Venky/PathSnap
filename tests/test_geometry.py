import unittest
import numpy as np
from src.data import Inputs
from src.geometry import Geometry,candidates
from src.alignment import align

def fixture():
    nodes=np.array([[10.,50.],[10.002,50.],[10.004,50.]])
    arcs=np.array([[0,0],[0,1],[1,0],[1,2],[2,1]])
    track=np.array([[10.0004,50.,0],[10.0016,50.,1],[10.001,50.,70],[10.0004,50.,71]])
    return Geometry(Inputs(track,nodes,arcs))

class GeometryChecks(unittest.TestCase):
    def test_long_segment_and_original_arc_ids(self):
        g=fixture()
        c=candidates(g,5,4)
        self.assertEqual(set(c.ids[0][c.ids[0]>=0]),{1,2})
        self.assertAlmostEqual(c.frac[0,np.where(c.ids[0]==1)[0][0]],.2,places=4)
        self.assertTrue(140<g.length[1]<145)
        self.assertEqual(c.X[2,0,7],0)
    def test_occurrences_and_gap(self):
        g=fixture()
        route=np.array([1,2,1,3])
        truth,occ,d,amb,blocks=align(g,route)
        np.testing.assert_array_equal(truth,route[occ])
        self.assertTrue(blocks[2])
        for sub in np.split(occ,np.flatnonzero(blocks)[1:]): self.assertTrue((np.diff(sub)>=0).all())
    def test_repeat_after_loop_and_distant_label(self):
        nodes=np.array([[0.,0.],[.0003,0.],[.0003,.0003],[.01,.01]])
        arcs=np.array([[0,1],[1,2],[2,0]])
        points=np.array([[.00005,0,0],[.00025,0,1],[.0003,.00015,2],[.00015,.00015,3],[.00002,0,4],[.00023,0,5],[.01,.01,100]])
        g=Geometry(Inputs(points,nodes,arcs))
        truth,occ,*_=align(g,[0,1,2,0])
        self.assertEqual((occ[0],occ[5]),(0,3))
        self.assertEqual(truth[-1],-1)
        self.assertTrue(np.all(candidates(g).ids[-1]==-1))
