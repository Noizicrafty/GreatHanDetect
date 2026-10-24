import unittest
import numpy as np
from greathan.config import ROOT
from greathan.contracts import Observation,Person
from greathan.rules import SmokingRule


def observation(object_box=(98,54,106,58),with_pose=True):
    points=np.zeros((17,2),np.float32)
    points[0]=[100,50];points[1]=[95,45];points[2]=[105,45]
    points[5]=[80,90];points[6]=[120,90];points[7]=[75,120];points[8]=[125,120]
    points[9]=[100,56];points[10]=[125,150]
    person=Person([60,30,140,250],.95,points if with_pose else None,np.ones(17),1)
    return Observation([person],[('cigarette',.9,np.array(object_box))])


class SmokingRuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (ROOT/'vendor/smoking-detection/smokedet/fusion.py').exists():
            raise unittest.SkipTest('Run python tools/bootstrap_source.py to test the pinned smoking rule')

    def test_strong_mouth_object_requires_window_then_positive(self):
        rule=SmokingRule({'window_s':6},ROOT/'vendor/smoking-detection')
        states=[]
        for t in np.arange(0,7,.5):states.append(rule.evaluate(observation(),(280,200),float(t))[0].positive)
        self.assertIsNone(states[0]);self.assertTrue(states[-1])

    def test_object_far_from_head_and_hand_is_not_smoking(self):
        rule=SmokingRule({'window_s':6},ROOT/'vendor/smoking-detection')
        for t in np.arange(0,7,.5):result=rule.evaluate(observation((0,200,10,210)),(280,200),float(t))[0]
        self.assertFalse(result.positive)

    def test_missing_pose_is_unknown(self):
        rule=SmokingRule({},ROOT/'vendor/smoking-detection')
        self.assertIsNone(rule.evaluate(observation(with_pose=False),(280,200),0)[0].positive)
