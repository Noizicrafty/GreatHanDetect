from __future__ import annotations
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import threading
import time
import uuid

import cv2
import numpy as np

from .config import ROOT, public_config, validate_config
from .contracts import config_hash
from .inference import Vision
from .metrics import Metrics
from .rules import EpisodeManager, SmokingRule, ZoneRule, RuleResult
from .tracking import Tracker
from .video import VideoSource
from .backends import BackendError
from .capabilities import demo_events


class Pipeline:
    def __init__(self, store, data_dir, vision_factory=Vision):
        self.store = store
        self.directory = Path(data_dir).resolve()
        self.directory.mkdir(parents=True,exist_ok=True)
        self.evidence = self.directory/"evidence"
        self.evidence.mkdir(exist_ok=True)
        self.vision_factory = vision_factory
        self.lock = threading.RLock()
        self.thread = None
        self.stop_signal = threading.Event()
        self.jpeg = None
        self.status = {"state":"idle","message":"Источник не запущен","metrics":{}}

    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps(self.status))

    def _status(self, **values):
        with self.lock:
            self.status.update(values)

    def start(self, config):
        config = validate_config(public_config(config))
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise ValueError("Обработка уже запущена; дождитесь остановки")
            self.stop_signal = threading.Event()
            self.jpeg = None
            run_id = uuid.uuid4().hex
            self.status = {"state":"starting","message":"Открытие источника и моделей", "run_id":run_id,
                           "backend":config["backend"],"scenario":config["scenario"],"camera_id":config["camera"]["id"],
                           "demo_additional_enabled":bool(config.get('demo_additional',{}).get('enabled')),
                           "source_mode":config["camera"].get("mode","offline"),"metrics":{}}
            self.thread = threading.Thread(target=self._run,args=(config,run_id),name="safety-pipeline",daemon=True)
            self.thread.start()
            return run_id

    def stop(self, wait=False):
        self.stop_signal.set()
        if self.thread and self.thread.is_alive():
            self._status(state="stopping",message="Ожидание завершения текущего вызова")
            if wait:
                self.thread.join(timeout=125)

    @staticmethod
    def _render(frame, observation, zones, active_tracks, label):
        frame = frame.copy()
        h,w = frame.shape[:2]
        for zone in zones:
            points = np.asarray([[round(x*w),round(y*h)] for x,y in zone["polygon"]],dtype=np.int32)
            cv2.polylines(frame,[points],True,(0,170,255),2)
            cv2.putText(frame,zone["id"],tuple(points[0]),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,170,255),2)
        for person in observation.persons:
            x1,y1,x2,y2 = map(round,person.box)
            color = (0,0,255) if person.track_id in active_tracks else (80,220,80)
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,f'ID {person.track_id} {person.confidence:.2f}',(x1,max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,2)
            if person.keypoints is not None and person.keypoint_confidence is not None:
                for a,b in [(0,1),(0,2),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]:
                    if person.keypoint_confidence[a]>=.3 and person.keypoint_confidence[b]>=.3:
                        cv2.line(frame,tuple(map(round,person.keypoints[a])),tuple(map(round,person.keypoints[b])),(0,210,240),2)
                for point,confidence in zip(person.keypoints,person.keypoint_confidence):
                    if confidence>=.3:cv2.circle(frame,tuple(map(round,point)),3,(0,210,240),-1)
        for name,score,box in observation.objects:
            if name == "person":
                continue
            x1,y1,x2,y2 = map(round,box)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,180,0),2)
            cv2.putText(frame,f'{name} {score:.2f}',(x1,max(20,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,180,0),1)
        cv2.rectangle(frame,(0,0),(w,28),(25,25,25),-1)
        cv2.putText(frame,label,(8,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        return frame

    def _save_evidence(self, event_id, frame):
        ok,encoded = cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,92])
        if not ok:
            raise OSError("Snapshot encoding failed")
        destination = self.evidence/(event_id+".jpg")
        temporary = destination.with_suffix(".tmp")
        temporary.write_bytes(encoded.tobytes())
        temporary.replace(destination)
        return "/evidence/"+destination.name

    def _run(self, config, run_id):
        source = vision = metrics = writer = None
        camera_id = config["camera"]["id"]
        manager = EpisodeManager(**{k:config["rules"][k] for k in ("confirm_s","clear_s","missing_s","min_samples")})
        anchor = datetime.now(timezone.utc)
        def at(seconds):
            return (anchor+timedelta(seconds=seconds)).isoformat(timespec="milliseconds")
        last_source_s = 0.0
        end_status = "error"
        summary = {}
        demo_children={}
        try:
            self.store.start_run(run_id,camera_id,config["backend"],public_config(config))
            metrics = Metrics(self.directory/"runs"/run_id)
            vision = self.vision_factory(config)
            source = VideoSource(config["camera"],self.stop_signal)
            anchor=datetime.now(timezone.utc)
            rule = ZoneRule(config["zones"]) if config["scenario"] == "danger_zone" else SmokingRule(config.get("smoking",{}),ROOT/"vendor/smoking-detection")
            tracker = Tracker(ttl_s=config["rules"]["missing_s"]+1)
            next_analysis = 0.0
            failures = 0
            self._status(state="running",message="Обработка видео",model_version=vision.version,source_fps=source.fps,source_fps_estimated=source.fps_estimated)
            while not self.stop_signal.is_set():
                item = source.read()
                if item is None:
                    end_status = "stopped" if self.stop_signal.is_set() else "completed"
                    break
                frame_id,source_s,frame = item
                last_source_s = source_s
                if source_s+1e-6 < next_analysis:
                    source.dropped += 1
                    continue
                next_analysis = source_s+1/float(config["analysis_fps"])
                started = time.perf_counter()
                try:
                    observation = vision.observe(frame,frame_id)
                    failures = 0
                except Exception as exc:
                    failures += 1
                    # Exception type is sufficient here; source URLs and broker details must not enter the journal.
                    message = str(exc) if isinstance(exc,(BackendError,ValueError)) else f"Inference failed ({type(exc).__name__}); verify model manifest and connection"
                    self.store.diagnostic("inference_failed",message,run_id,camera_id)
                    self._status(state="degraded",message=message,consecutive_failures=failures)
                    unknowns = [RuleResult(k[0],k[1],k[2],None) for k in manager.episodes]
                    for action,key,episode,_ in manager.update(unknowns,source_s):
                        if episode.event_id and action == "lost":
                            self.store.update_event(episode.event_id,at(episode.last_positive),episode.last_positive,at(source_s),"lost")
                            for child_id in demo_children.get(episode.event_id,[]):
                                self.store.update_event(child_id,at(episode.last_positive),episode.last_positive,at(source_s),'lost')
                    if failures >= 3:
                        raise RuntimeError("Three consecutive inference failures") from None
                    self.stop_signal.wait(min(failures,3))
                    continue
                inference_done = time.perf_counter()
                tracker.update(observation.persons,source_s)
                results = rule.evaluate(observation,frame.shape,source_s)
                actions = manager.update(results,source_s)
                confirmation_wall = time.perf_counter()
                active = {k[0] for k,e in manager.episodes.items() if e.state == "active"}
                label = f'{config["backend"]} | {config["scenario"]} | {camera_id} | video {source_s:.2f}s'
                if config["backend"] == "replay":
                    label = "REPLAY TEST / NO NEURAL INFERENCE | "+label
                rendered = self._render(frame,observation,config["zones"],active,label)
                created = 0
                latencies = []
                for action,key,episode,result in actions:
                    tid,zone,event_type = key
                    if action == "create":
                        uri = self._save_evidence(episode.event_id,rendered)
                        latency = time.perf_counter()-confirmation_wall
                        event = {
                            "id":episode.event_id,"episode_id":episode.event_id,"run_id":run_id,
                            "camera_id":camera_id,"zone_id":zone,"event_type":event_type,"track_id":tid,
                            "observed_start":at(episode.started),"confirmed_at":at(episode.confirmed),
                            "last_seen":at(episode.last_positive),"ended_at":None,
                            "source_start":episode.started,"source_confirmed":episode.confirmed,"source_last":episode.last_positive,
                            "frame_id":frame_id,"snapshot_uri":uri,"lifecycle_state":"active",
                            "backend_kind":config["backend"],"model_version":vision.version,
                            "config_version":config_hash(public_config(config)),"confidence":result.confidence,
                            "reason":result.reason,"evidence_latency_s":latency,
                        }
                        if self.store.create_event(event):
                            created += 1
                            latencies.append(time.perf_counter()-confirmation_wall)
                            self.store.evidence_latency(episode.event_id,latencies[-1])
                            children=[]
                            for child in demo_events(event,config):
                                illustrated=rendered.copy()
                                cv2.rectangle(illustrated,(0,0),(illustrated.shape[1],28),(100,40,100),-1)
                                cv2.putText(illustrated,'DEMO / NO EXTRA MODEL | '+child['event_type'],(8,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
                                child['snapshot_uri']=self._save_evidence(child['id'],illustrated)
                                if self.store.create_event(child):children.append(child['id'])
                            demo_children[episode.event_id]=children
                    elif episode.event_id:
                        ended = at(source_s) if action != "update" else None
                        self.store.update_event(episode.event_id,at(episode.last_positive),episode.last_positive,ended,
                                                "lost" if action == "lost" else "ended" if ended else "active")
                        for child_id in demo_children.get(episode.event_id,[]):
                            self.store.update_event(child_id,at(episode.last_positive),episode.last_positive,ended,
                                                    "lost" if action=='lost' else 'ended' if ended else 'active')
                ok,encoded = cv2.imencode(".jpg",rendered,[cv2.IMWRITE_JPEG_QUALITY,80])
                if ok:
                    with self.lock:
                        self.jpeg = encoded.tobytes()
                if config.get("save_annotated",True):
                    if writer is None:
                        h,w = rendered.shape[:2]
                        writer = cv2.VideoWriter(str(metrics.directory/"annotated.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),float(config["analysis_fps"]),(w,h))
                        if not writer.isOpened():
                            raise OSError("Cannot create annotated video")
                    writer.write(rendered)
                finished = time.perf_counter()
                metrics.record({"frame_id":frame_id,"source_s":source_s,"wall_s":finished-metrics.started,
                                "inference_s":inference_done-started,"pipeline_s":finished-started,
                                "persons":len(observation.persons),"created_events":created,
                                "evidence_latency_s":max(latencies) if latencies else None},observation.timings,latencies)
                self._status(state="running",message="Обработка видео",source_s=source_s,frame_id=frame_id,
                             persons=len(observation.persons),source_time_basis=source.time_basis,metrics=metrics.summary(source.dropped))
            else:
                end_status = "stopped"
        except Exception as exc:
            # Known validation errors contain no credentials because configuration rejects embedded auth.
            message = str(exc) if isinstance(exc,(ValueError,OSError,RuntimeError)) else type(exc).__name__
            self.store.diagnostic("pipeline_failed",message,run_id,camera_id)
            self._status(state="error",message=message)
        finally:
            self.stop_signal.set()
            for _,key,episode,_ in manager.finish():
                self.store.update_event(episode.event_id,at(episode.last_positive),episode.last_positive,at(last_source_s),
                                        "interrupted" if end_status == "error" else "ended")
                for child_id in demo_children.get(episode.event_id,[]):
                    self.store.update_event(child_id,at(episode.last_positive),episode.last_positive,at(last_source_s),
                                            'interrupted' if end_status=='error' else 'ended')
            if source:
                if not source.close():
                    end_status="error"
                    self.store.diagnostic("capture_shutdown_timeout","Capture reader did not stop within timeout",run_id,camera_id)
                    self._status(state="error",message="Capture reader shutdown timed out")
            if writer:
                writer.release()
            if vision:
                vision.close()
            if metrics:
                summary = metrics.close(source.dropped if source else 0)
            self.store.finish_run(run_id,end_status,summary)
            if end_status != "error":
                self._status(state=end_status,message="Видео завершено" if end_status == "completed" else "Обработка остановлена",metrics=summary)
