from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import uuid
import sqlite3
from unittest.mock import patch

from greathan.config import load_config, public_config, validate_config, ROOT
from greathan.pipeline import Pipeline
from greathan.server import Application, make_handler
from greathan.store import EventStore
from tools.make_app_demo import generate


def event(**changes):
    id=uuid.uuid4().hex
    item={"id":id,"episode_id":id,"run_id":"run","camera_id":"cam","zone_id":"zone",
          "event_type":"danger_zone","track_id":1,"observed_start":"2026-10-24T10:00:00.000+00:00",
          "confirmed_at":"2026-10-24T10:00:02.000+00:00","last_seen":"2026-10-24T10:00:03.000+00:00",
          "ended_at":None,"source_start":0.,"source_confirmed":2.,"source_last":3.,"frame_id":21,
          "snapshot_uri":"/evidence/"+id+".jpg","lifecycle_state":"active","backend_kind":"replay",
          "model_version":"qa","config_version":"qa","confidence":.9,"reason":"test","evidence_latency_s":.01}
    item.update(changes)
    return item


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/"journal.db"
        self.store=EventStore(self.path)

    def test_event_and_notification_are_idempotent(self):
        item=event()
        self.assertTrue(self.store.create_event(item))
        self.assertFalse(self.store.create_event(item))
        self.assertEqual(self.store.events()["total"],1)
        notes=self.store.notifications()
        self.assertEqual(len(notes["items"]),1)
        self.assertEqual(self.store.notifications(notes["cursor"])["items"],[])

    def test_invalid_event_does_not_silently_disappear_or_notify(self):
        item=event();del item['snapshot_uri']
        with self.assertRaises(sqlite3.IntegrityError):self.store.create_event(item)
        self.assertEqual(self.store.events()['total'],0)
        self.assertEqual(self.store.notifications()['items'],[])

    def test_four_filters_and_combination(self):
        self.store.create_event(event())
        self.store.create_event(event(camera_id="other",zone_id="other",event_type="smoking"))
        filters={"camera_id":"cam","zone_id":"zone","event_type":"danger_zone",
                 "from":"2026-10-24T12:59:00+03:00","to":"2026-10-24T13:01:00+03:00"}
        self.assertEqual(self.store.events(filters)["total"],1)
        for key in ["camera_id","zone_id","event_type"]:
            self.assertEqual(self.store.events({key:"absent"})["total"],0)
        self.assertEqual(self.store.events({"from":"2026-10-25T00:00:00Z"})["total"],0)
        with self.assertRaises(ValueError):self.store.events({"from":"2026-10-24T10:00:00"})

    def test_restart_preserves_journal_and_marks_interrupted(self):
        item=event();self.store.create_event(item)
        restarted=EventStore(self.path)
        self.assertEqual(restarted.get(item["id"])["lifecycle_state"],"interrupted")
        self.assertEqual(restarted.events()["total"],1)
        self.assertEqual(len(restarted.notifications()["items"]),1)

    def test_review_and_sql_input_are_safe(self):
        item=event();self.store.create_event(item)
        self.assertTrue(self.store.review(item["id"],"false_positive","operator note"))
        self.assertEqual(self.store.get(item["id"])["review_status"],"false_positive")
        self.assertEqual(self.store.events({"camera_id":"' OR 1=1 --"})["total"],0)
        with self.assertRaises(ValueError):self.store.review(item["id"],"invalid")

    def test_display_ack_is_idempotent(self):
        item=event();self.store.create_event(item)
        self.assertTrue(self.store.displayed(item['id']))
        first=self.store.delivery_metrics()
        self.assertTrue(self.store.displayed(item['id']))
        second=self.store.delivery_metrics()
        self.assertEqual(first['items'],second['items'])
        self.assertEqual(second['n'],1)

    def test_notification_pagination_never_skips_backlog(self):
        for _ in range(105):self.store.create_event(event())
        first=self.store.notifications(0)
        second=self.store.notifications(first["cursor"])
        self.assertEqual(len(first["items"]),100)
        self.assertEqual(len(second["items"]),5)


class PipelineIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.config=load_config(generate(Path(cls.temp.name)/"demo"))
        cls.config["camera"]["mode"]="offline"

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_video_to_two_episodes_evidence_and_metrics(self):
        directory=Path(self.temp.name)/"run"
        store=EventStore(directory/"journal.db")
        pipeline=Pipeline(store,directory)
        pipeline.start(self.config)
        pipeline.thread.join(timeout=20)
        self.assertFalse(pipeline.thread.is_alive())
        self.assertEqual(pipeline.snapshot()["state"],"completed",store.diagnostics())
        events=store.events()["items"]
        self.assertEqual(len(events),2)
        self.assertEqual(len(store.notifications()["items"]),2)
        for item in events:
            self.assertEqual(item["backend_kind"],"replay")
            self.assertIn(item["lifecycle_state"],("ended","lost"))
            self.assertTrue((directory/"evidence"/Path(item["snapshot_uri"]).name).is_file())
        summary=pipeline.snapshot()["metrics"]
        self.assertEqual(summary["new_events"],2)
        self.assertGreater(summary["processed_frames"],90)
        self.assertTrue((directory/"runs"/pipeline.snapshot()["run_id"]/"frames.csv").is_file())
        self.assertIsNotNone(pipeline.jpeg)

    def test_bad_source_reports_error_without_losing_journal(self):
        directory=Path(self.temp.name)/"bad"
        store=EventStore(directory/"journal.db")
        store.create_event(event())
        pipeline=Pipeline(store,directory)
        config=json.loads(json.dumps(self.config));config["camera"]["source"]="not-a-file.mp4"
        pipeline.start(config);pipeline.thread.join(timeout=10)
        self.assertEqual(pipeline.snapshot()["state"],"error")
        self.assertEqual(store.events()["total"],1)
        self.assertEqual(store.diagnostics()[0]["code"],"pipeline_failed")

    def test_failed_inference_never_creates_safe_scene(self):
        class BrokenVision:
            version="broken-test"
            def __init__(self,config):pass
            def observe(self,*args):raise TimeoutError("secret should not leak")
            def close(self):pass
        directory=Path(self.temp.name)/"failed-inference"
        store=EventStore(directory/"journal.db")
        pipeline=Pipeline(store,directory,vision_factory=BrokenVision)
        pipeline.start(self.config);pipeline.thread.join(timeout=15)
        self.assertEqual(pipeline.snapshot()["state"],"error")
        self.assertEqual(store.events()["total"],0)
        self.assertGreaterEqual(len(store.diagnostics()),3)
        self.assertNotIn("secret",json.dumps(store.diagnostics()))

    def test_demo_events_follow_parent_but_do_not_enter_real_metrics(self):
        from greathan.rules import RuleResult
        class AlwaysSmoking:
            def __init__(self,*args):pass
            def evaluate(self,observation,*args):
                return [RuleResult(p.track_id,'s','smoking',True,.9,'QA rule fixture') for p in observation.persons]
        directory=Path(self.temp.name)/'demo-children'
        store=EventStore(directory/'journal.db');pipeline=Pipeline(store,directory)
        config=json.loads(json.dumps(self.config));config['scenario']='smoking'
        config['demo_additional']={'enabled':True,'scenarios':['helmet','fall']}
        with patch('greathan.pipeline.SmokingRule',AlwaysSmoking):
            pipeline.start(config);pipeline.thread.join(timeout=20)
        self.assertFalse(pipeline.thread.is_alive())
        self.assertEqual(pipeline.snapshot()['state'],'completed',store.diagnostics())
        events=store.events()['items'];real=[e for e in events if e['event_type']=='smoking'];demo=[e for e in events if e['backend_kind']=='demo_simulated']
        self.assertGreater(len(real),0);self.assertEqual(len(demo),2*len(real))
        self.assertEqual(pipeline.snapshot()['metrics']['new_events'],len(real))
        self.assertTrue(all(e['lifecycle_state'] in ('lost','ended') for e in events))
        for item in demo:store.displayed(item['id'])
        self.assertEqual(store.delivery_metrics()['n'],0)


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        config=public_config(load_config(ROOT/"configs/application.yaml"))
        self.app=Application(config,self.temp.name)
        self.server=ThreadingHTTPServer(("127.0.0.1",0),make_handler(self.app))
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.base=f"http://127.0.0.1:{self.server.server_port}"
        self.addCleanup(self.shutdown)

    def shutdown(self):
        self.server.shutdown();self.server.server_close();self.thread.join()
        self.app.close()

    def test_second_application_cannot_recover_live_journal(self):
        item=event();self.app.store.create_event(item)
        with self.assertRaises(RuntimeError):Application(self.app.config,self.temp.name)
        self.assertEqual(self.app.store.get(item['id'])['lifecycle_state'],'active')

    def request(self,path,body=None,headers=None):
        return urlopen(Request(self.base+path,data=json.dumps(body).encode() if body is not None else None,
                       headers={"Content-Type":"application/json",**(headers or {})}),timeout=3)

    def test_ui_status_and_empty_history(self):
        with self.request("/") as r:self.assertIn("Контроль безопасности",r.read().decode())
        with self.request("/api/status") as r:self.assertEqual(json.load(r)["state"],"idle")
        with self.request("/api/notifications?after=latest") as r:self.assertEqual(json.load(r)["items"],[])

    def test_cross_origin_writes_blocked(self):
        with self.assertRaises(HTTPError) as error:self.request("/api/stop",{},headers={"Origin":"https://attacker.example"})
        self.assertEqual(error.exception.code,403)

    def test_invalid_host_is_rejected(self):
        with self.assertRaises(HTTPError) as error:self.request('/api/config',headers={'Host':'attacker.example'})
        self.assertEqual(error.exception.code,403)

    def test_event_review_and_export(self):
        item=event();self.app.store.create_event(item)
        with self.request("/api/events/"+item["id"]+"/review",{"status":"confirmed"}) as r:self.assertTrue(json.load(r)["ok"])
        with self.request("/api/export.csv?camera_id=cam") as r:self.assertIn(item["id"],r.read().decode("utf-8-sig"))
        with self.request("/api/events?zone_id=zone") as r:self.assertEqual(json.load(r)["total"],1)

    def test_path_traversal_does_not_read_files(self):
        with self.assertRaises(HTTPError) as error:self.request("/evidence/../../configs/application.yaml")
        self.assertEqual(error.exception.code,404)

    def test_invalid_config_rejected(self):
        config=public_config(self.app.config)
        config["zones"][0]["polygon"]=[[0,0],[0,0],[0,0]]
        with self.assertRaises(HTTPError) as error:self.request("/api/config",config)
        self.assertEqual(error.exception.code,400)
