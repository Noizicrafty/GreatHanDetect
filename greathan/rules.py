"""Scenario rules and event episodes are independent from the inference transport."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import uuid


def inside_polygon(point, polygon):
    """Ray casting including points on the boundary; normalized coordinates."""
    x, y = point
    inside = False
    for (ax, ay), (bx, by) in zip(polygon, polygon[1:] + polygon[:1]):
        cross = (x-ax)*(by-ay) - (y-ay)*(bx-ax)
        if abs(cross) < 1e-9 and min(ax,bx)-1e-9 <= x <= max(ax,bx)+1e-9 and min(ay,by)-1e-9 <= y <= max(ay,by)+1e-9:
            return True
        if (ay > y) != (by > y) and x < (bx-ax)*(y-ay)/(by-ay)+ax:
            inside = not inside
    return inside


@dataclass
class RuleResult:
    track_id: int
    zone_id: str
    event_type: str
    positive: bool | None
    confidence: float = 0.0
    reason: str = ""


class ZoneRule:
    def __init__(self, zones):
        self.zones = zones

    def evaluate(self, observation, shape, timestamp):
        height, width = shape[:2]
        return [RuleResult(p.track_id, z["id"], "danger_zone",
                           inside_polygon(((p.box[0]+p.box[2])/2/width, p.box[3]/height), z["polygon"]),
                           p.confidence, "person footpoint in configured zone")
                for p in observation.persons for z in self.zones]


class SmokingRule:
    """Uses the pinned upstream feature/fusion heuristics, without its batch-only event writer."""
    def __init__(self, config, source_root):
        import sys
        from pathlib import Path
        import yaml
        source_root = Path(source_root)
        if not (source_root / "smokedet/fusion.py").is_file():
            raise ValueError("Smoking requires: python tools/bootstrap_source.py")
        sys.path.insert(0, str(source_root))
        from smokedet.features import FeatureParams, frame_features, sequence_features, feature_names, BASE_FEATURES
        from smokedet.fusion import FusionParams, fuse, window_object_score
        from smokedet.motion import MotionParams, motion_score
        from smokedet.taxonomy import load_taxonomy
        original = yaml.safe_load((source_root/"configs/pipeline.yaml").read_text(encoding="utf-8"))
        tax = load_taxonomy(source_root/"configs/taxonomy.yaml")
        self.frame_features, self.sequence_features = frame_features, sequence_features
        self.fuse, self.window_object_score, self.motion_score = fuse, window_object_score, motion_score
        self.tax, self.names = tax, feature_names(tax.classes)
        self.object_columns = [len(BASE_FEATURES)+tax.classes.index(c) for c in tax.smoking_objects]
        self.distance_index = self.names.index("d_min")
        self.feature_params = FeatureParams(kp_conf_thr=original.get("kp_conf_thr", .3), **{**original.get("features", {}),**config.get("features",{})})
        self.fusion_params = FusionParams(**{**original.get("fusion", {}),**config.get("fusion",{})})
        self.motion_params = MotionParams(**{**original.get("motion", {}),**config.get("motion",{})})
        self.window_s = float(config.get("window_s", 6))
        self.require_object = bool(config.get("require_object",True))
        self.max_gap_s = float(config.get("max_gap_s",3))
        self.buffers = {}

    def evaluate(self, observation, shape, timestamp):
        results = []
        for tid in list(self.buffers):
            if self.buffers[tid] and timestamp-self.buffers[tid][-1][0] > 20:
                del self.buffers[tid]
        for p in observation.persons:
            if p.keypoints is None:
                results.append(RuleResult(p.track_id, "smoking_area", "smoking", None, reason="pose unavailable"))
                continue
            features = self.frame_features(p.keypoints, p.keypoint_confidence, p.box, observation.objects,
                                           self.tax.classes, self.feature_params)
            buf = self.buffers.setdefault(p.track_id, deque(maxlen=1200))
            buf.append((timestamp, features))
            while len(buf) > 1 and timestamp-buf[0][0] > 20:
                buf.popleft()
            span = timestamp-buf[0][0]
            # A long unobserved interval must not be interpolated into a smoking motion.
            if len(buf) < 4 or span < self.window_s or any(b[0]-a[0] > self.max_gap_s for a,b in zip(buf,list(buf)[1:])):
                results.append(RuleResult(p.track_id, "smoking_area", "smoking", None, reason="collecting motion window"))
                continue
            import numpy as np
            # Heuristics expect uniform sampling. Resample by actual source time, not processing FPS.
            fps = 10.0
            times = np.arange(max(buf[0][0], timestamp-20), timestamp+1e-6, 1/fps)
            source_times = [x[0] for x in buf]
            source_features = self.sequence_features([x[1] for x in buf],len(self.names))
            sequence = np.column_stack([np.interp(times,source_times,source_features[:,i]) for i in range(len(self.names))])
            window = max(3,round(self.window_s*fps))
            obj = self.window_object_score(sequence[-window:,self.object_columns], self.fusion_params.obj_percentile)
            motion = self.motion_score(sequence[:,self.distance_index],fps,window,self.motion_params)
            decision, score, reason = self.fuse(obj,motion,self.fusion_params)
            if self.require_object and obj < self.fusion_params.obj_thr:
                decision=False
                reason="insufficient smoking object evidence"
            results.append(RuleResult(p.track_id,"smoking_area","smoking",bool(decision),float(score),str(reason)))
        return results


@dataclass
class Episode:
    started: float
    last_positive: float
    last_observed: float
    confirmed: float | None = None
    negative_since: float | None = None
    event_id: str | None = None
    state: str = "pending"


class EpisodeManager:
    """One creation per episode. Missing evidence is unknown, never a safe negative."""
    def __init__(self, confirm_s=1.5, clear_s=2.0, missing_s=3.0, min_samples=3):
        self.confirm_s, self.clear_s, self.missing_s = confirm_s, clear_s, missing_s
        self.min_samples = min_samples
        self.episodes = {}
        self.samples = {}

    def update(self, results, timestamp):
        for key in list(self.episodes):
            if self.episodes[key].state=="ended" and timestamp-self.episodes[key].last_observed>max(30,self.missing_s*2):
                del self.episodes[key]
                self.samples.pop(key,None)
        actions = []
        observed = set()
        unknown = set()
        for result in results:
            key = (result.track_id, result.zone_id, result.event_type)
            observed.add(key)
            episode = self.episodes.get(key)
            if result.positive is True:
                if episode is None or episode.state == "ended":
                    episode = self.episodes[key] = Episode(timestamp,timestamp,timestamp)
                    self.samples[key] = 0
                # Gaps before confirmation do not count as continuous positive evidence.
                if episode.state == "pending" and (episode.negative_since is not None or timestamp-episode.last_positive > self.missing_s):
                    episode.started = timestamp
                    self.samples[key] = 0
                episode.last_positive = episode.last_observed = timestamp
                episode.negative_since = None
                self.samples[key] = self.samples.get(key,0)+1
                if episode.state == "pending" and timestamp-episode.started >= self.confirm_s and self.samples[key] >= self.min_samples:
                    episode.state, episode.confirmed, episode.event_id = "active", timestamp, uuid.uuid4().hex
                    actions.append(("create",key,episode,result))
                elif episode.state == "active":
                    actions.append(("update",key,episode,result))
            elif episode is not None and episode.state != "ended":
                episode.last_observed = timestamp
                if result.positive is False:
                    if episode.state == "pending":
                        episode.state = "ended"
                    else:
                        if episode.negative_since is None:
                            episode.negative_since = timestamp
                        if timestamp-episode.negative_since >= self.clear_s:
                            episode.state = "ended"
                            actions.append(("end",key,episode,result))
                else:
                    unknown.add(key)
                    if episode.state == "pending":
                        episode.state = "ended"
        for key, episode in self.episodes.items():
            if episode.state=='pending' and key not in observed:
                episode.state='ended'
            if episode.state != "ended" and timestamp-episode.last_positive > self.missing_s and (key not in observed or key in unknown):
                episode.state = "ended"
                if episode.event_id:
                    actions.append(("lost",key,episode,None))
        return actions

    def finish(self):
        actions = []
        for key, episode in self.episodes.items():
            if episode.state == "active":
                episode.state = "ended"
                actions.append(("end",key,episode,None))
        return actions
