"""CPU tail for the ONNX Model Zoo Tiny YOLOv2 VOC raw grid, batch 1."""
import numpy as np
from tools.common import unletterbox_xyxy
from tools.decoder import nms

VOC_CLASSES = ["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow",
               "diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]
ANCHORS = np.array([[1.08,1.19],[3.42,4.41],[6.63,11.38],[9.42,5.11],[16.62,10.52]],np.float32)


def decode_tiny_yolo(raw, meta, size=416, confidence=.3, iou=.5):
    if raw.shape != (1,125,13,13):
        raise ValueError(f"Tiny YOLOv2 expected (1,125,13,13), got {raw.shape}")
    values = raw[0].reshape(5,25,13,13).transpose(0,2,3,1)
    sigmoid = lambda x: 1/(1+np.exp(-np.clip(x,-80,80)))
    grid_y,grid_x = np.mgrid[:13,:13]
    cx = (sigmoid(values[...,0])+grid_x)*size/13
    cy = (sigmoid(values[...,1])+grid_y)*size/13
    width = np.exp(np.clip(values[...,2],-15,15))*ANCHORS[:,None,None,0]*size/13
    height = np.exp(np.clip(values[...,3],-15,15))*ANCHORS[:,None,None,1]*size/13
    logits = values[...,5:]
    probabilities = np.exp(logits-logits.max(axis=-1,keepdims=True))
    probabilities /= probabilities.sum(axis=-1,keepdims=True)
    class_ids = probabilities.argmax(axis=-1).reshape(-1)
    scores = (sigmoid(values[...,4])*probabilities.max(axis=-1)).reshape(-1)
    boxes = np.stack([cx-width/2,cy-height/2,cx+width/2,cy+height/2],axis=-1).reshape(-1,4).astype(np.float32)
    selected = []
    for cls in np.unique(class_ids[scores>=confidence]):
        ids = np.flatnonzero(class_ids==cls)
        selected.extend(ids[k] for k in nms(boxes[ids],scores[ids],confidence,iou))
    if not selected:
        return []
    mapped = unletterbox_xyxy(boxes[selected],meta)
    return [(VOC_CLASSES[class_ids[index]],float(scores[index]),mapped[j]) for j,index in enumerate(selected)]
