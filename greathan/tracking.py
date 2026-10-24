"""Small deterministic CPU tracker with one-to-one assignment and short occlusion memory."""
from __future__ import annotations
from dataclasses import dataclass
import math

from .contracts import Person


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = max(0, a[2]-a[0])*max(0, a[3]-a[1]) + max(0, b[2]-b[0])*max(0, b[3]-b[1]) - intersection
    return intersection/union if union else 0.0


@dataclass
class Track:
    person: Person
    last_seen: float


class Tracker:
    def __init__(self, ttl_s=3.0, iou_threshold=.2):
        self.ttl_s, self.threshold = ttl_s, iou_threshold
        self.tracks = {}
        self.next_id = 1

    def update(self, persons, timestamp):
        self.tracks = {k: v for k, v in self.tracks.items() if timestamp-v.last_seen <= self.ttl_s}
        candidates = []
        for index, person in enumerate(persons):
            for tid, track in self.tracks.items():
                overlap = box_iou(person.box, track.person.box)
                if overlap >= self.threshold:
                    candidates.append((overlap, tid, index))
        used_tracks, used_people = set(), set()
        for _, tid, index in sorted(candidates, reverse=True):
            if tid not in used_tracks and index not in used_people:
                persons[index].track_id = tid
                used_tracks.add(tid)
                used_people.add(index)
        for index, person in enumerate(persons):
            if index not in used_people:
                person.track_id = self.next_id
                self.next_id += 1
            self.tracks[person.track_id] = Track(person, timestamp)
        return persons
