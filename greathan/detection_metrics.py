"""Object detection AP50 with confidence-ordered, one-to-one class-aware matching."""
from __future__ import annotations
import numpy as np
from .tracking import box_iou


def detection_report(records,class_names,confidence=.25,iou=.5):
    per_class=[]
    for cls,name in enumerate(class_names):
        truth={record['image']:[box for label,box in record['truth'] if label==cls] for record in records}
        count=sum(map(len,truth.values()))
        predictions=sorted([(float(score),record['image'],box) for record in records for label,score,box in record['predictions'] if label==cls],key=lambda p:p[0],reverse=True)
        matched={image:set() for image in truth}
        tp=[];fp=[];scores=[]
        for score,image,box in predictions:
            candidates=[(box_iou(box,target),index) for index,target in enumerate(truth[image]) if index not in matched[image]]
            best=max(candidates,default=(0,None))
            success=best[0]>=iou
            if success:matched[image].add(best[1])
            tp.append(int(success));fp.append(int(not success));scores.append(score)
        tps=np.cumsum(tp);fps=np.cumsum(fp)
        recall=tps/count if count else np.zeros(len(tp))
        precision=tps/np.maximum(tps+fps,1)
        ap=None
        if count:
            # 101-point interpolated AP, explicitly AP50 when iou=.5 (not COCO mAP50:95).
            ap=float(np.mean([max(precision[recall>=level],default=0) for level in np.linspace(0,1,101)]))
        selected=np.asarray(scores)>=confidence
        op_tp=sum(np.asarray(tp)[selected]);op_fp=sum(np.asarray(fp)[selected])
        per_class.append({'class_id':cls,'name':name,'truth_objects':count,'predicted_objects':len(predictions),
                          'ap50':ap,'tp':int(op_tp),'fp':int(op_fp),'fn':int(count-op_tp),
                          'precision':float(op_tp/(op_tp+op_fp)) if op_tp+op_fp else None,
                          'recall':float(op_tp/count) if count else None})
    ap_values=[c['ap50'] for c in per_class if c['ap50'] is not None]
    return {'images':len(records),'failed_images':sum(bool(r.get('error')) for r in records),'operating_confidence':confidence,
            'matching_iou':iou,'map50':float(np.mean(ap_values)) if ap_values else None,'per_class':per_class,
            'scope':'Object-level image AP50, 101-point interpolation, classes without truth excluded from mean. Failed requests count as missing detections. Not an event metric or field validation.'}
