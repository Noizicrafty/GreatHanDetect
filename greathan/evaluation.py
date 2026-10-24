"""Event-level one-to-one matching; this does not turn synthetic QA into model quality evidence."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def overlap(predicted, truth, tolerance_s=0):
    for key in ("camera_id","event_type","zone_id"):
        if predicted.get(key)!=truth.get(key):
            return 0.0
    start=max(float(predicted["start_s"]),float(truth["start_s"])-tolerance_s)
    end=min(float(predicted["end_s"]),float(truth["end_s"])+tolerance_s)
    intersection=max(0,end-start)
    union=max(float(predicted["end_s"]),float(truth["end_s"])+tolerance_s)-min(float(predicted["start_s"]),float(truth["start_s"])-tolerance_s)
    return intersection/union if union>0 else 0.0


def evaluate(predictions, truths, duration_s, minimum_iou=.1, tolerance_s=0):
    def validate(items):
        for item in items:
            if float(item["start_s"])<0 or float(item["end_s"])<=float(item["start_s"]):
                raise ValueError("Every event requires 0 <= start_s < end_s")
            for key in ("camera_id","event_type","zone_id"):
                if not item.get(key):raise ValueError("Every event requires "+key)
    validate(predictions);validate(truths)
    if duration_s<=0:raise ValueError("Observation duration must be positive")
    if not 0<minimum_iou<=1 or tolerance_s<0:
        raise ValueError("Require 0 < minimum_iou <= 1 and tolerance_s >= 0")
    if any(float(x["end_s"])>duration_s for x in predictions+truths):
        raise ValueError("Events must fit inside declared observation duration")
    candidates={p:sorted([t for t in range(len(truths)) if overlap(predictions[p],truths[t],tolerance_s)>=minimum_iou],
                         key=lambda t:overlap(predictions[p],truths[t],tolerance_s),reverse=True) for p in range(len(predictions))}
    matched_truth={}
    def assign(p,visited):
        for t in candidates[p]:
            if t in visited:continue
            visited.add(t)
            if t not in matched_truth or assign(matched_truth[t],visited):
                matched_truth[t]=p
                return True
        return False
    for p in sorted(candidates,key=lambda p:len(candidates[p])):
        assign(p,set())
    matched_predictions=set(matched_truth.values())
    tp=len(matched_truth);fp=len(predictions)-tp;fn=len(truths)-tp
    duplicates=sum(p not in matched_predictions and any(t in matched_truth for t in candidates[p]) for p in candidates)
    return {"tp":tp,"fp":fp,"fn":fn,"duplicates":duplicates,
            "precision":tp/(tp+fp) if tp+fp else None,"recall":tp/(tp+fn) if tp+fn else None,
            "duration_s":duration_s,"false_alarms_per_observed_hour":fp/(duration_s/3600),
            "minimum_temporal_iou":minimum_iou,"tolerance_s":tolerance_s,
            "matches":[{"truth_index":t,"prediction_index":p,"iou":overlap(predictions[p],truths[t],tolerance_s)} for t,p in sorted(matched_truth.items())],
            "unmatched_prediction_indices":[p for p in range(len(predictions)) if p not in matched_predictions],
            "unmatched_truth_indices":[t for t in range(len(truths)) if t not in matched_truth],
            "limitation":"Event matching only. Report source dataset, independent episodes and real observation duration; short QA clips do not establish field accuracy."}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions",required=True)
    parser.add_argument("--truth",required=True)
    parser.add_argument("--duration-s",type=float,required=True)
    parser.add_argument("--out",required=True)
    parser.add_argument("--minimum-iou",type=float,default=.1)
    parser.add_argument("--tolerance-s",type=float,default=0)
    args=parser.parse_args()
    report=evaluate(json.loads(Path(args.predictions).read_text()),json.loads(Path(args.truth).read_text()),args.duration_s,args.minimum_iou,args.tolerance_s)
    Path(args.out).write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ("matches",)},indent=2))


if __name__=="__main__":main()
