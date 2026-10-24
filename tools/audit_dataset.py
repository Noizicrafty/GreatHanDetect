"""Audit YOLO annotations and remove exact/near cross-split duplicates from training."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import yaml


def fingerprint(path):
    image=cv2.imread(str(path))
    if image is None:raise ValueError('Cannot decode '+str(path))
    gray=cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(32,32)).astype(np.float32)
    def phash(array):
        values=cv2.dct(array)[:8,:8].reshape(-1)
        bits=values>np.median(values[1:])
        return sum(int(bit)<<i for i,bit in enumerate(bits))
    return {'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'phash':min(phash(gray),phash(gray[:,::-1].copy())),'shape':list(image.shape[:2])}


def audit(root,out,distance=4):
    root=Path(root).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    metadata=yaml.safe_load((root/'data.yaml').read_text(encoding='utf-8'))
    names=metadata['names']
    if isinstance(names,dict):names=[names[i] for i in sorted(names)]
    records={};invalid=[];counts={}
    for split in ['train','valid','test']:
        files=sorted((root/split/'images').glob('*'))
        images=[p for p in files if p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.webp')]
        rows=[]
        def inspect(path):
            label=root/split/'labels'/(path.stem+'.txt')
            if not label.is_file():raise ValueError('Missing annotation '+str(label))
            boxes=[]
            for line in label.read_text(encoding='utf-8').splitlines():
                if not line.strip():continue
                numbers=list(map(float,line.split()))
                if len(numbers)!=5 or not all(np.isfinite(numbers)):
                    raise ValueError('Invalid detection annotation '+str(label))
                cls=int(numbers[0]);xywh=numbers[1:]
                if numbers[0]!=cls or not 0<=cls<len(names) or any(v<0 or v>1 for v in xywh) or min(xywh[2:])<=0:
                    raise ValueError('Annotation outside YOLO ranges '+str(label))
                boxes.append(numbers)
            return {'path':str(path),'split':split,'boxes':boxes,**fingerprint(path)}
        with ThreadPoolExecutor(max_workers=8) as pool:
            for path,result in zip(images,pool.map(lambda p:inspect_safely(inspect,p),images)):
                if isinstance(result,str):invalid.append({'path':str(path),'error':result})
                else:rows.append(result)
        records[split]=rows
        counts[split]={'images':len(images),'valid_images':len(rows),'boxes':dict(Counter(names[int(b[0])] for row in rows for b in row['boxes']))}
    # Keep the test set. Remove related validation images, then remove related training images.
    buckets=defaultdict(list)
    sha_index={}
    # Five bands guarantee at least one unchanged band for Hamming distance <= 4.
    def bands(value):
        return [(index,(value>>(13*index))&((1<<(12 if index==4 else 13))-1)) for index in range(5)]
    def add(row):
        sha_index[row['sha256']]=row
        for key in bands(row['phash']):buckets[key].append(row)
    def match(row):
        if row['sha256'] in sha_index:return sha_index[row['sha256']],'exact'
        candidates={}
        for key in bands(row['phash']):
            for candidate in buckets.get(key,[]):candidates[candidate['path']]=candidate
        for candidate in candidates.values():
            if (row['phash']^candidate['phash']).bit_count()<=distance:return candidate,'near_phash'
        return None,None
    removed=[];kept={'test':records['test']}
    for row in records['test']:add(row)
    for split in ['valid','train']:
        kept[split]=[]
        for row in records[split]:
            other,reason=match(row)
            if other:
                removed.append({'path':row['path'],'split':split,'matches':other['path'],'reason':reason})
            else:kept[split].append(row)
        # Do not deduplicate within training: supplied augmentation is intentional.
        if split=='valid':
            for row in kept[split]:add(row)
    normalized={'path':str(root),'nc':len(names),'names':names}
    for split,key in [('train','train'),('valid','val'),('test','test')]:
        path=out/(split+'.txt')
        path.write_text('\n'.join(row['path'] for row in kept[split])+'\n',encoding='utf-8')
        normalized[key]=str(path)
    (out/'data.yaml').write_text(yaml.safe_dump(normalized,allow_unicode=True,sort_keys=False),encoding='utf-8')
    summary={'source':str(root),'metadata':metadata,'counts_before':counts,'kept':{k:len(v) for k,v in kept.items()},
             'boxes_after':{k:dict(Counter(names[int(b[0])] for row in rows for b in row['boxes'])) for k,rows in kept.items()},
             'invalid':invalid,'removed':removed,'phash_hamming_threshold':distance,
             'limitation':'Conservative exact/perceptual duplicate screening; does not prove independence of video sources or actors. No keypoint annotations in this detection dataset.'}
    (out/'audit.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    with (out/'fingerprints.jsonl').open('w',encoding='utf-8') as stream:
        for rows in records.values():
            for row in rows:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    return summary


def inspect_safely(function,path):
    try:return function(path)
    except Exception as exc:return str(exc)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--near-distance',type=int,default=4)
    args=parser.parse_args()
    summary=audit(args.root,args.out,args.near_distance)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('removed','invalid','metadata')},indent=2,ensure_ascii=False))
    print('Invalid:',len(summary['invalid']),'cross-split related removed:',len(summary['removed']))
