"""CPU postprocessing for un-NMS'ed Ultralytics ONNX object/pose heads."""
from __future__ import annotations
import cv2
import numpy as np
try:
    from .common import unletterbox_xyxy
except ImportError:  # Direct execution of legacy tools.
    from common import unletterbox_xyxy

def tensor_nxc(raw,features):
    a=np.asarray(raw)
    if a.ndim!=3 or a.shape[0]!=1: raise ValueError(f'Unsupported shape {a.shape}, expected [1,{features},N] or [1,N,{features}]')
    a=a[0]
    if a.shape[0]==features: return a.T
    if a.shape[1]==features: return a
    raise ValueError(f'Unsupported shape {raw.shape}: expected {features} channels')

def xywh_to_xyxy(x):
    y=x.copy()
    y[:,0]=x[:,0]-x[:,2]/2; y[:,1]=x[:,1]-x[:,3]/2
    y[:,2]=x[:,0]+x[:,2]/2; y[:,3]=x[:,1]+x[:,3]/2
    return y

def nms(boxes,scores,conf,iou,max_det=100):
    inds=np.flatnonzero(scores>=conf)
    if len(inds)==0:return []
    b=boxes[inds]
    wh=np.column_stack([b[:,0],b[:,1],b[:,2]-b[:,0],b[:,3]-b[:,1]]).tolist()
    candidates=cv2.dnn.NMSBoxes(wh,scores[inds].tolist(),float(conf),float(iou),top_k=1000)
    if len(candidates)==0:return []
    return inds[np.asarray(candidates).reshape(-1)[:max_det]].tolist()

def decode_objects(raw,meta,classes,conf=0.15,iou=0.5):
    arr=tensor_nxc(raw,4+len(classes))
    boxes=xywh_to_xyxy(arr[:,:4])
    cls_ids=arr[:,4:].argmax(axis=1)
    scores=arr[:,4:].max(axis=1)
    picked=[]
    for c in np.unique(cls_ids[scores>=conf]):
        ids=np.flatnonzero(cls_ids==c)
        keep=nms(boxes[ids],scores[ids],conf,iou)
        picked.extend(ids[k] for k in keep)
    boxes=unletterbox_xyxy(boxes[np.array(picked,dtype=int)],meta) if picked else np.empty((0,4))
    return [(str(classes[cls_ids[k]]),float(scores[k]),boxes[j]) for j,k in enumerate(picked)]

def decode_pose(raw,meta,conf=0.3,iou=0.5):
    n_keypoints=17; f=5+n_keypoints*3
    arr=tensor_nxc(raw,f)
    xyxy=xywh_to_xyxy(arr[:,:4]); scores=arr[:,4]
    keep=nms(xyxy,scores,conf,iou)
    if not keep:return []
    mapped=unletterbox_xyxy(xyxy[np.array(keep,dtype=int)],meta)
    gain,dx,dy,w,h=meta
    result=[]
    for i,k in enumerate(keep):
        pts=arr[k,5:].reshape(n_keypoints,3).copy()
        pts[:,0]=np.clip((pts[:,0]-dx)/gain,0,w)
        pts[:,1]=np.clip((pts[:,1]-dy)/gain,0,h)
        result.append((mapped[i],float(scores[k]),pts[:,:2],pts[:,2]))
    return result
