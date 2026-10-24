import unittest
from greathan.contracts import Person, Observation
from greathan.rules import EpisodeManager, RuleResult, ZoneRule, inside_polygon
from greathan.tracking import Tracker


def result(positive=True,tid=1,zone="zone-a"):
    return RuleResult(tid,zone,"danger_zone",positive,.9,"test")


class EpisodeTests(unittest.TestCase):
    def manager(self):
        return EpisodeManager(confirm_s=1,clear_s=2,missing_s=3,min_samples=3)

    def test_long_violation_creates_once(self):
        manager=self.manager()
        actions=[]
        for tick in range(121):
            actions.extend(manager.update([result()],tick*.5))
        self.assertEqual(sum(a[0]=="create" for a in actions),1)
        self.assertEqual(len(manager.finish()),1)

    def test_short_positive_is_not_confirmed(self):
        manager=self.manager()
        manager.update([result()],0)
        manager.update([result()],.5)
        self.assertEqual(manager.update([result(False)],.8),[])
        self.assertEqual(manager.finish(),[])

    def test_negative_before_confirmation_resets_clock(self):
        manager=self.manager()
        manager.update([result()],0)
        manager.update([result(False)],.8)
        self.assertEqual(manager.update([result()],1),[])
        manager.update([result()],1.5)
        self.assertEqual(manager.update([result()],2)[0][0],"create")

    def test_short_occlusion_does_not_duplicate(self):
        manager=self.manager()
        for t in [0,.5,1]:manager.update([result()],t)
        self.assertEqual(manager.update([],2),[])
        self.assertEqual(manager.update([result()],2.5)[0][0],"update")

    def test_two_people_and_zones_are_independent(self):
        manager=self.manager()
        actions=[]
        for t in [0,.5,1]:
            actions.extend(manager.update([result(tid=1),result(tid=2),result(tid=1,zone="b")],t))
        self.assertEqual(sum(a[0]=="create" for a in actions),3)
        self.assertEqual(len({a[2].event_id for a in actions}),3)

    def test_clear_then_new_episode(self):
        manager=self.manager()
        ids=[]
        for t in [0,.5,1]:
            ids.extend(a[2].event_id for a in manager.update([result()],t) if a[0]=="create")
        manager.update([result(False)],2)
        self.assertEqual(manager.update([result(False)],4)[0][0],"end")
        for t in [5,5.5,6]:
            ids.extend(a[2].event_id for a in manager.update([result()],t) if a[0]=="create")
        self.assertEqual(len(set(ids)),2)

    def test_unknown_is_not_negative_and_event_can_be_lost(self):
        manager=self.manager()
        for t in [0,.5,1]:manager.update([result()],t)
        self.assertEqual(manager.update([result(None)],2),[])
        self.assertEqual(manager.update([result(None)],4.1)[0][0],"lost")

    def test_pending_unknown_requires_new_confirmation(self):
        manager=self.manager()
        manager.update([result()],0)
        manager.update([result(None)],.5)
        self.assertEqual(manager.update([result()],1),[])
        self.assertEqual(manager.update([result()],1.5),[])
        self.assertEqual(manager.update([result()],2)[0][0],"create")

    def test_pending_missing_observation_does_not_accumulate_confirmation(self):
        manager=self.manager()
        manager.update([result()],0)
        manager.update([],0.5)
        self.assertEqual(manager.update([result()],1),[])
        self.assertEqual(manager.update([result()],1.5),[])
        self.assertEqual(manager.update([result()],2)[0][0],'create')

    def test_zone_uses_footpoint_and_includes_boundary(self):
        zone={"id":"z","polygon":[[.5,.1],[.9,.1],[.9,.9],[.5,.9]]}
        rule=ZoneRule([zone])
        people=[Person([40,5,60,90],.9,track_id=1),Person([0,0,40,80],.9,track_id=2)]
        self.assertEqual([r.positive for r in rule.evaluate(Observation(people),[100,100],0)],[True,False])
        self.assertFalse(inside_polygon((.2,.2),zone["polygon"]))

    def test_tracker_retains_id_through_short_occlusion(self):
        tracker=Tracker(ttl_s=3)
        person=lambda:Person([0,0,100,200],.9)
        first=tracker.update([person()],0)[0].track_id
        tracker.update([],1)
        self.assertEqual(tracker.update([person()],2)[0].track_id,first)
        self.assertNotEqual(tracker.update([person()],6)[0].track_id,first)

    def test_tracker_assignments_are_one_to_one(self):
        tracker=Tracker()
        tracker.update([Person([0,0,100,200],.9)],0)
        people=tracker.update([Person([0,0,100,200],.9),Person([5,0,105,200],.9)],.2)
        self.assertEqual(len({p.track_id for p in people}),2)
