import unittest
from greathan.capabilities import catalogue,demo_events


class CapabilityTests(unittest.TestCase):
    def test_extra_models_are_never_reported_as_connected(self):
        extras=[c for c in catalogue() if c['id'] in ('helmet','vest','fall')]
        self.assertEqual(len(extras),3)
        self.assertTrue(all(c['state']=='model_not_connected' and c['mode']=='demo_only' for c in extras))

    def test_demo_children_have_separate_identity_and_explicit_provenance(self):
        parent={'id':'a','episode_id':'a','event_type':'smoking','backend_kind':'linq_amqp','confidence':.9}
        config={'demo_additional':{'enabled':True,'scenarios':['helmet','fall']}}
        children=demo_events(parent,config)
        self.assertEqual(len(children),2)
        self.assertEqual(len({x['id'] for x in children}),2)
        self.assertTrue(all(c['backend_kind']=='demo_simulated' and c['confidence']==0 for c in children))
        self.assertEqual(parent['backend_kind'],'linq_amqp')
        self.assertEqual(demo_events(parent,{}),[])
        self.assertEqual(demo_events({**parent,'event_type':'danger_zone'},config),[])
