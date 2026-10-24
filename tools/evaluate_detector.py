"""Evaluate the exact application decoder over a fixed dataset split, on CPU or real LinQ."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import yaml
from greathan.config import load_config
from greathan.inference import Vision
from greathan.detection_metrics import detection_report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--data',required=True)
    parser.add_argument('--model',default='detector');parser.add_argument('--split',choices=['train','val','test'],default='val')
    parser.add_argument('--limit',type=int,default=0);parser.add_argument('--confidence',type=float,default=.25)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    config=load_config(args.config)
    if config['backend']=='replay':raise SystemExit('Quality evaluation requires a neural backend')
    dataset=yaml.safe_load(Path(args.data).read_text(encoding='utf-8'))
    names=dataset['names'];names=[names[i] for i in sorted(names)] if isinstance(names,dict) else names
    model=config['models'][args.model]
    if model['class_names']!=names:raise SystemExit('Model classes and dataset classes differ; provide an explicit curated mapping')
    paths=Path(dataset[args.split]).read_text(encoding='utf-8').splitlines()
    paths=[Path(p) for p in paths if p.strip()]
    if args.limit:
        paths=random.Random(42).sample(paths,min(args.limit,len(paths)))
    output=Path(args.out);output.mkdir(parents=True,exist_ok=True)
    model['confidence']=.001
    vision=Vision(config)
    records=[]
    raw=(output/'predictions.jsonl').open('w',encoding='utf-8')
    try:
        for index,path in enumerate(paths):
            frame=cv2.imread(str(path))
            if frame is None:raise ValueError('Cannot decode evaluation image: '+str(path))
            height,width=frame.shape[:2]
            label=path.parent.parent/'labels'/(path.stem+'.txt')
            truth=[]
            for line in label.read_text().splitlines():
                if not line.strip():continue
                cls,cx,cy,w,h=map(float,line.split())
                truth.append([int(cls),[(cx-w/2)*width,(cy-h/2)*height,(cx+w/2)*width,(cy+h/2)*height]])
            record={'image':path.name,'truth':truth,'predictions':[]}
            started=time.perf_counter()
            try:
                _,objects,timing=vision._model(args.model,frame)
                record['predictions']=[[names.index(name),float(score),list(map(float,box))] for name,score,box in objects]
                record['timing']=timing
            except Exception as exc:
                record['error']=type(exc).__name__
            record['wall_s']=time.perf_counter()-started
            records.append(record);raw.write(json.dumps(record)+'\n');raw.flush()
            print(f'{index+1}/{len(paths)} {path.name} {record["wall_s"]:.3f}s',flush=True)
        report=detection_report(records,names,args.confidence)
        report.update(backend=config['backend'],model_version=vision.version,split=args.split,selection_seed=42 if args.limit else None,
                      evaluated_list_sha256=hashlib.sha256('\n'.join(str(p) for p in paths).encode()).hexdigest(),
                      inference_confidence=.001,nms_iou=model.get('iou',.5))
        (output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2))
    finally:
        raw.close();vision.close()


if __name__=='__main__':main()
