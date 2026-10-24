"""Make representative float32 preprocessing tensors for TPU PTQ.
Uses images or sampled video frames; does NOT train or validate model labels.
"""
import argparse
import json
import random
from pathlib import Path
import numpy as np
from common import dest, project_cfg, letterbox

EXTS={'.jpg','.jpeg','.png','.bmp','.webp'}

def image_paths(paths):
    for val in paths:
        p=Path(val)
        if p.is_dir():
            for f in sorted(p.rglob('*')):
                if f.suffix.lower() in EXTS: yield f
        elif p.suffix.lower() in EXTS: yield p


def frames(paths, video, max_frames, seed):
    import cv2
    items=list(image_paths(paths))
    random.Random(seed).shuffle(items)
    quota=max_frames
    for file in items[:quota]:
        frame=cv2.imread(str(file))
        if frame is not None:
            yield frame, str(file)
    if video:
        cap=cv2.VideoCapture(str(video))
        if not cap.isOpened(): raise SystemExit('Cannot read video: '+str(video))
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        try:
            stride=max(1,total//max_frames) if total>0 else 30
            index=0
            while index < (total if total>0 else 100000):
                if total>0: cap.set(cv2.CAP_PROP_POS_FRAMES,index)
                ok,frame=cap.read()
                if not ok: break
                yield frame, f'{video}:frame{index}'
                index+=stride
        finally: cap.release()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--part',choices=['detector','pose'],required=True)
    ap.add_argument('--images',nargs='*',default=[])
    ap.add_argument('--video',default=None)
    ap.add_argument('--limit',type=int,default=24)
    ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args()
    if not args.images and not args.video: raise SystemExit('Pass --images <folder> or --video <mp4>')
    import onnx
    cfg=project_cfg()['models'][args.part]
    model=onnx.load(str(dest(cfg['onnx'])))
    if len(model.graph.input)!=1: raise SystemExit('Expected exactly one ONNX input')
    name=model.graph.input[0].name
    arrays=[];names=[]
    for frame,origin in frames(args.images,args.video,args.limit,args.seed):
        arr,_=letterbox(frame,int(cfg['input_size']))
        arrays.append(arr); names.append(origin)
        if len(arrays)>=args.limit: break
    if not arrays: raise SystemExit('No usable images found')
    data=np.concatenate(arrays,axis=0)
    output=dest(cfg['calibration']);output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,**{name:data})
    (output.with_suffix('.json')).write_text(json.dumps({'part':args.part,'input':name,
        'samples':len(names),'origins':names,'shape':list(data.shape),
        'normalization':'RGB/255, NCHW, YOLO letterbox with 114 gray'},indent=2,ensure_ascii=False),encoding='utf-8')
    print('CALIBRATION:',output,'samples:',len(names),'shape:',data.shape)
    if len(names)<16: print('WARNING: recommend at least 16-100 representative frames incl. low-quality CCTV views.')

if __name__=='__main__':main()
