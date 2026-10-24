"""Run an application model on an image; retain raw results, hashes and transport timings."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from greathan.config import load_config
from greathan.inference import Vision
from greathan.pipeline import Pipeline


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--image',required=True)
    parser.add_argument('--out',default='results/model_check')
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--requests',type=int,default=20)
    args=parser.parse_args()
    config=load_config(args.config)
    if config['backend']=='replay':raise SystemExit('Use a real inference backend for a model check')
    if not 0<=args.warmup<=100 or not 1<=args.requests<=10000:
        raise SystemExit('Require 0..100 warmup and 1..10000 measured requests')
    frame=cv2.imread(args.image)
    if frame is None:raise SystemExit('Cannot read image')
    directory=Path(args.out);directory.mkdir(parents=True,exist_ok=True)
    vision=Vision(config)
    records=[]
    (directory/'requests.jsonl').write_text('',encoding='utf-8')
    try:
        for index in range(args.warmup+args.requests):
            start=time.perf_counter()
            observation=vision.observe(frame,index)
            elapsed=time.perf_counter()-start
            record={'index':index,'warmup':index<args.warmup,'total_s':elapsed,'timings':observation.timings,
                    'persons':[{'box':p.box,'confidence':p.confidence} for p in observation.persons]}
            records.append(record)
            with (directory/'requests.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(record)+'\n')
            print(index,'warmup' if record['warmup'] else 'measured',f'{elapsed:.4f}s',len(observation.persons),'person detections',flush=True)
        values=[r['total_s'] for r in records if not r['warmup']]
        summary={'backend':config['backend'],'scenario':config['scenario'],'model_version':vision.version,
                 'warmup_n':args.warmup,'measured_n':len(values),'p50_s':float(np.percentile(values,50)),
                 'p95_s':float(np.percentile(values,95)),
                 'scope':'Vision.observe: preprocessing, transport, worker, CPU decode; no event confirmation or UI',
                 'image':Path(args.image).name,'quality_validation':False}
        (directory/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        rendered=Pipeline._render(frame,observation,[],set(),config['backend']+' | actual inference | image demonstration')
        cv2.imwrite(str(directory/'detections.jpg'),rendered)
        print(json.dumps(summary,indent=2))
    finally:
        vision.close()


if __name__=='__main__':main()
