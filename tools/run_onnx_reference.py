"""ONNX Runtime diagnostic, draws detected objects and people; no temporal behavior inference."""
import argparse
import cv2
import onnxruntime as ort
from common import letterbox,project_cfg,dest
from decoder import decode_objects,decode_pose

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--video',required=True)
    ap.add_argument('--max-frames',type=int,default=200)
    ap.add_argument('--show',action='store_true')
    args=ap.parse_args()
    cfg=project_cfg()['models']
    sess={part:ort.InferenceSession(str(dest(spec['onnx'])),providers=['CPUExecutionProvider'])
          for part,spec in cfg.items()}
    source=int(args.video) if args.video.isdecimal() else args.video
    cap=cv2.VideoCapture(source)
    if not cap.isOpened():raise SystemExit('Cannot open camera or video')
    output=dest('results/onnx_reference.mp4');output.parent.mkdir(parents=True,exist_ok=True)
    writer=None
    try:
        for step in range(args.max_frames):
            ok,frame=cap.read()
            if not ok:break
            preds={}
            for part,spec in cfg.items():
                sample,meta=letterbox(frame,spec['input_size'])
                sn=sess[part];inp=sn.get_inputs()[0].name
                results=sn.run(None,{inp:sample})
                if len(results)!=1:raise SystemExit(f'{part}: expected one output')
                preds[part]=(results[0],meta)
            objects=decode_objects(*preds['detector'],cfg['detector']['class_names'],.15,.5)
            people=decode_pose(*preds['pose'],.3,.5)
            for name,conf,b in objects:
                x1,y1,x2,y2=map(int,b)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,140,255),2)
                cv2.putText(frame,f'{name} {conf:.2f}',(x1,max(0,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,140,255),2)
            for b,conf,pts,pconf in people:
                x1,y1,x2,y2=map(int,b)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,180,0),2)
            if writer is None:
                h,w=frame.shape[:2]
                fps=cap.get(cv2.CAP_PROP_FPS) or 10
                writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
            writer.write(frame)
            if args.show:
                cv2.imshow('ONNX reference',frame)
                if cv2.waitKey(1)&255==ord('q'):break
    finally:
        cap.release()
        if writer:writer.release()
        if args.show:cv2.destroyAllWindows()
    print('Output:',output)

if __name__=='__main__':main()
