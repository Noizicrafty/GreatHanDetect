"""Select a reproducible, class-covered calibration subset strictly from the training list."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import yaml
from greathan.config import load_config
from tools.common import letterbox


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--data',required=True)
    parser.add_argument('--model',default='detector');parser.add_argument('--count',type=int,default=32)
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--out',required=True)
    args=parser.parse_args()
    if not 4<=args.count<=256:raise SystemExit('Calibration count must be 4..256')
    dataset=yaml.safe_load(Path(args.data).read_text(encoding='utf-8'))
    paths=list(dict.fromkeys(Path(dataset['train']).read_text(encoding='utf-8').splitlines()))
    paths=[Path(p) for p in paths if p.strip()]
    forbidden={Path(p).resolve() for key in ['val','test'] for p in Path(dataset[key]).read_text(encoding='utf-8').splitlines() if p.strip()}
    if any(p.resolve() in forbidden for p in paths):raise SystemExit('Train/holdout list overlap; audit the dataset first')
    classes=defaultdict(list)
    for path in paths:
        label=path.parent.parent/'labels'/(path.stem+'.txt')
        for cls in {int(line.split()[0]) for line in label.read_text().splitlines() if line.strip()}:classes[cls].append(path)
    rng=random.Random(args.seed);selected=[]
    for cls in sorted(classes,key=lambda c:len(classes[c])):
        candidates=classes[cls].copy();rng.shuffle(candidates)
        for path in candidates:
            if path not in selected:selected.append(path)
            if sum(path in classes[cls] for path in selected)>=max(1,args.count//len(classes)):break
    candidates=paths.copy();rng.shuffle(candidates)
    for path in candidates:
        if len(selected)>=args.count:break
        if path not in selected:selected.append(path)
    selected=selected[:args.count]
    model=load_config(args.config)['models'][args.model]
    arrays=[];sources=[]
    for path in selected:
        frame=cv2.imread(str(path))
        if frame is None:raise ValueError('Cannot read calibration image')
        arr,_=letterbox(frame,int(model['input_size']));arr*=float(model.get('input_scale',1))
        arrays.append(arr[0]);sources.append({'image':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    tensor=np.asarray(arrays,dtype=np.float32)
    np.savez_compressed(out,**{next(iter(model['inputs'])):tensor})
    out.with_suffix('.json').write_text(json.dumps({'split':'train','seed':args.seed,'shape':list(tensor.shape),'sources':sources,'preprocessing':'application letterbox RGB NCHW with declared input scale'},indent=2),encoding='utf-8')
    print(out,tensor.shape,flush=True)


if __name__=='__main__':main()
