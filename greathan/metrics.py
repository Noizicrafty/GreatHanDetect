from __future__ import annotations
import csv
import json
from pathlib import Path
import time
import numpy as np


class Metrics:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.started = time.perf_counter()
        self.rows = []
        self.event_latencies=[]
        self.stream = (self.directory/"frames.csv").open("w",encoding="utf-8",newline="")
        self.writer = csv.DictWriter(self.stream,fieldnames=["frame_id","source_s","wall_s","inference_s","pipeline_s","persons","created_events","evidence_latency_s"])
        self.writer.writeheader()
        self.raw = (self.directory/"inference.jsonl").open("w",encoding="utf-8")

    def record(self,row,timings,event_latencies=()):
        self.rows.append(row)
        self.event_latencies.extend(event_latencies)
        self.writer.writerow(row)
        self.stream.flush()
        self.raw.write(json.dumps({"frame_id":row["frame_id"],"source_s":row["source_s"],"timings":timings})+"\n")
        self.raw.flush()

    def summary(self, dropped=0):
        elapsed = time.perf_counter()-self.started
        def percentiles(key):
            values = [r[key] for r in self.rows if r.get(key) is not None]
            return {"p50_s":float(np.percentile(values,50)),"p95_s":float(np.percentile(values,95)),"n":len(values)} if values else {"n":0}
        return {"processed_frames":len(self.rows),"elapsed_wall_s":elapsed,
                "processing_fps":len(self.rows)/elapsed if elapsed else 0,
                "source_span_s":self.rows[-1]["source_s"] if self.rows else 0,
                "dropped_frames":dropped,"inference":percentiles("inference_s"),
                "pipeline":percentiles("pipeline_s"),"confirmation_to_journal":(
                    {"n":len(self.event_latencies),"p50_s":float(np.percentile(self.event_latencies,50)),"p95_s":float(np.percentile(self.event_latencies,95))}
                    if self.event_latencies else {"n":0}),
                "new_events":sum(r["created_events"] for r in self.rows),
                "timing_scope":"inference includes preprocessing/transport/CPU decoder; confirmation_to_journal excludes temporal confirmation and UI polling"}

    def close(self,dropped=0):
        summary = self.summary(dropped)
        self.stream.close()
        self.raw.close()
        (self.directory/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
        return summary
