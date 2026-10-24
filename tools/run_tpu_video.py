"""Experimental two-model TPU video runtime; clyv fusion stays on CPU.

Compatibility requires BOTH TPU programs to have Ultralytics raw non-NMS head tensors.
No claim of vendor certification: the SDK lists neither YOLO-World nor YOLO11-Pose.
"""
import argparse
import json
import sys
from collections import deque
from pathlib import Path
import cv2
import numpy as np
from common import dest,ensure_clyv_path,project_cfg,letterbox
from decoder import decode_objects,decode_pose
from tpu_backend import TPUDualSession

SRC=ensure_clyv_path()
from smokedet.features import BASE_FEATURES,feature_names,sequence_features,FeatureParams,frame_features
from smokedet.fusion import EventAggregator,FusionParams,fuse,window_object_score
from smokedet.motion import MotionParams,motion_score
from smokedet.taxonomy import load_taxonomy
from smokedet.utils import load_yaml

class IoUTracker:
    """Minimal CPU IoU tracker; not a ByteTrack replacement."""
    def __init__(self,threshold=.3,max_lost=15):
        self.threshold=threshold;self.max_lost=max_lost;self.next_id=1;self.active={}
    @staticmethod
    def iou(a,b):
        x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3])
        inter=max(0,x2-x1)*max(0,y2-y1)
        aarea=max(0,a[2]-a[0])*max(0,a[3]-a[1]);barea=max(0,b[2]-b[0])*max(0,b[3]-b[1])
        return float(inter/(aarea+barea-inter+1e-6))
    def assign(self,poses):
        proposals=sorted(((self.iou(box,state[0]),i,tid) for i,(box,*_) in enumerate(poses)
                         for tid,state in self.active.items()),reverse=True)
        ids={};used=set()
        for score,i,tid in proposals:
            if score<self.threshold:break
            if i in ids or tid in used:continue
            ids[i]=tid;used.add(tid)
        for i in range(len(poses)):
            if i not in ids:ids[i]=self.next_id;self.next_id+=1
        updated={tid:(box,0) for i,(box,*_) in enumerate(poses) for tid in [ids[i]]}
        for tid,(box,lost) in self.active.items():
            if tid not in updated and lost+1<=self.max_lost:updated[tid]=(box,lost+1)
        self.active=updated
        return ids


def single_output_names(path):
    import onnx
    graph=onnx.load(str(path)).graph
    if len(graph.input)!=1 or len(graph.output)!=1:
        raise ValueError('Expected a single-input/single-output, no-NMS Ultralytics ONNX. Export manually with --nms false.')
    return graph.input[0].name,graph.output[0].name


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--video',required=True,help='MP4 path, RTSP URL or 0 for webcam')
    ap.add_argument('--show',action='store_true')
    ap.add_argument('--out',default='results/tpu_pipeline')
    ap.add_argument('--max-frames',type=int,default=0)
    args=ap.parse_args()
    cfg=project_cfg();mc=cfg['models'];t=cfg['inference']
    original=load_yaml(SRC/'configs/pipeline.yaml')
    tax=load_taxonomy(SRC/'configs/taxonomy.yaml')
    names=feature_names(tax.classes);n_feat=len(names)
    obj_cols=[len(BASE_FEATURES)+tax.classes.index(c) for c in tax.smoking_objects]
    d_idx=names.index('d_min')
    feature_params=FeatureParams(kp_conf_thr=original.get('kp_conf_thr',.30),**original.get('features',{}))
    mparam=MotionParams(**original.get('motion',{}))
    fparam=FusionParams(**original.get('fusion',{}))
    tracker=IoUTracker(threshold=t['iou_tracking'])
    agg=EventAggregator(**original.get('events',{}))
    d_in,d_out=single_output_names(dest(mc['detector']['onnx']))
    p_in,p_out=single_output_names(dest(mc['pose']['onnx']))
    for part in ('detector','pose'):
        if not dest(mc[part]['tpu']).exists():raise SystemExit('Missing TPU program: '+str(dest(mc[part]['tpu'])))
    src=int(args.video) if args.video.isdecimal() else args.video
    capture=cv2.VideoCapture(src)
    if not capture.isOpened():raise SystemExit(f'Cannot read video/camera: {args.video}')
    fps=capture.get(cv2.CAP_PROP_FPS) or 30.0
    stride=max(1,round(fps/t['target_fps']))
    eff_fps=fps/stride
    window=max(1,round(eff_fps*t['window_s']))
    hop=max(1,round(eff_fps*t['hop_s']))
    max_missing=max(1,round(eff_fps*t['max_missing_s']))
    buffers={};last_seen={};last_scores={};step=frame_no=0
    out=dest(args.out);out.mkdir(parents=True,exist_ok=True)
    stem='camera' if isinstance(src,int) else Path(str(src).split('?')[0]).stem
    writer=None
    try:
        with TPUDualSession(dest(mc['detector']['tpu']),dest(mc['pose']['tpu']),d_in,p_in,d_out,p_out) as session:
            while True:
                ok,frame=capture.read()
                if not ok:break
                frame_no+=1
                if (frame_no-1)%stride:continue
                t_s=(frame_no-1)/fps
                arr_d,info_d=letterbox(frame,mc['detector']['input_size'])
                arr_p,info_p=letterbox(frame,mc['pose']['input_size'])
                objects=decode_objects(session.detect(arr_d),info_d,mc['detector']['class_names'],
                                       t['confidence_objects'],t['iou_nms'])
                poses=decode_pose(session.pose(arr_p),info_p,t['confidence_pose'],t['iou_nms'])
                ids=tracker.assign(poses)
                seen=set()
                for i,(box,confidence,keypoints,kconf) in enumerate(poses):
                    tid=ids[i];seen.add(tid)
                    feats=frame_features(keypoints,kconf,box,objects,tax.classes,feature_params)
                    buffers.setdefault(tid,deque(maxlen=max(window,round(eff_fps*20)))).append(feats)
                    last_seen[tid]=step
                for tid in list(buffers):
                    if tid in seen:continue
                    if step-last_seen[tid]>max_missing:
                        agg.close_track(tid,t_s);buffers.pop(tid,None);last_seen.pop(tid,None);last_scores.pop(tid,None)
                    else:buffers[tid].append(None)
                if step%hop==0:
                    for tid,buf in buffers.items():
                        if sum(v is not None for v in buf)<max(3,window//3):continue
                        seq=sequence_features(list(buf),n_feat)
                        obj=window_object_score(seq[-window:,obj_cols],fparam.obj_percentile)
                        mot=motion_score(seq[:,d_idx],eff_fps,window,mparam)
                        decision,score,reason=fuse(obj,mot,fparam)
                        agg.update(tid,t_s,decision,score,obj,mot,reason)
                        last_scores[tid]=(obj,mot)
                for name,score,box in objects:
                    x1,y1,x2,y2=np.asarray(box,dtype=int)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,140,255),2)
                    cv2.putText(frame,f'{name} {score:.2f}',(x1,max(18,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,140,255),2)
                for i,(box,confidence,_,_) in enumerate(poses):
                    tid=ids[i];active=agg.is_active(tid)
                    x1,y1,x2,y2=np.asarray(box,dtype=int)
                    color=(0,0,255) if active else (0,180,0)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                    scores=last_scores.get(tid,(0,0))
                    cv2.putText(frame,f'ID {tid}'+(' SMOKING' if active else '')+f' obj={scores[0]:.2f} motion={scores[1]:.2f}',
                                (x1,max(18,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,2)
                if writer is None:
                    h,w=frame.shape[:2]
                    writer=cv2.VideoWriter(str(out/f'{stem}_annotated.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),eff_fps,(w,h))
                writer.write(frame)
                if args.show:
                    cv2.imshow('TPU smoking detection (experimental)',frame)
                    if cv2.waitKey(1)&0xff==ord('q'):break
                step+=1
                if args.max_frames and step>=args.max_frames:break
    finally:
        capture.release()
        if writer:writer.release()
        if args.show:cv2.destroyAllWindows()
    events=agg.finish(frame_no/fps)
    (out/f'{stem}_events.json').write_text(json.dumps({'events':events,'source':args.video,
       'runtime':'TPU_H1M O1 (experimental)','motion_source':'clyv heuristic',
       'tracker':'CPU IoU (not ByteTrack)'},indent=2),encoding='utf-8')
    print('Saved video and event JSON in',out,'events:',len(events))

if __name__=='__main__':main()
