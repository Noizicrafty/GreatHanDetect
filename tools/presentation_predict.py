"""Inference-only worker for the local presentation demo. Never calls train()."""
import argparse, json, hashlib, os, sys, base64, io, time
from pathlib import Path
os.environ.setdefault('MPLBACKEND','Agg')

def stream(checkpoint):
    """One resident model, one JSON request/response per line. No frames on disk."""
    from contextlib import redirect_stdout
    with redirect_stdout(sys.stderr):
        import torch
        from ultralytics import YOLO
        from PIL import Image
        torch.set_num_threads(2)
        model=YOLO(checkpoint)
        metadata={'epochs':int(model.ckpt.get('epoch',-1))+1,'checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),'backend':'pytorch_cuda' if torch.cuda.is_available() else 'pytorch_cpu','training_started':False}
    for line in sys.stdin:
        try:
            image=Image.open(io.BytesIO(base64.b64decode(json.loads(line)['image'],validate=True))).convert('RGB')
            started=time.perf_counter()
            with redirect_stdout(sys.stderr):
                result=model.predict(source=image,imgsz=640,conf=.25,iou=.7,device=0 if torch.cuda.is_available() else 'cpu',verbose=False)[0]
            detections=[{'class_id':int(c),'class_name':model.names[int(c)],'confidence':float(s),'xyxy':[float(x) for x in b]} for b,s,c in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist(),result.boxes.cls.cpu().tolist())]
            reply={**metadata,'size':list(image.size),'detections':detections,'inference_ms':round((time.perf_counter()-started)*1000,1)}
        except Exception as e:
            print(f'Webcam inference failed: {type(e).__name__}: {e}',file=sys.stderr,flush=True)
            reply={'error':'Не удалось выполнить инференс кадра'}
        print(json.dumps(reply,ensure_ascii=False),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--image')
    p.add_argument('--output');p.add_argument('--result');p.add_argument('--stream',action='store_true')
    a=p.parse_args()
    if a.stream:return stream(a.checkpoint)
    if not all((a.image,a.output,a.result)):p.error('--image, --output and --result are required without --stream')
    import torch
    from ultralytics import YOLO
    from PIL import Image,ImageDraw,ImageFont
    model=YOLO(a.checkpoint)
    epoch=int(model.ckpt.get('epoch',-1))+1
    result=model.predict(source=a.image,imgsz=640,conf=.25,iou=.7,device=0 if torch.cuda.is_available() else 'cpu',verbose=False)[0]
    detections=[{'class_id':int(c),'class_name':model.names[int(c)],'confidence':float(s),'xyxy':[float(x) for x in b]} for b,s,c in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist(),result.boxes.cls.cpu().tolist())]
    image=Image.open(a.image).convert('RGB');draw=ImageDraw.Draw(image)
    font_path=next((x for x in ['C:/Windows/Fonts/segoeuib.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'] if Path(x).exists()),None)
    font=ImageFont.truetype(font_path,max(15,image.width//45)) if font_path else ImageFont.load_default()
    colors={0:'#F3B434',1:'#24B6EA',2:'#ED5B6C',3:'#A47CE5'}
    for d in sorted(detections,key=lambda d:d['class_id']!=1):
        x1,y1,x2,y2=d['xyxy'];color=colors.get(d['class_id'],'#24B6EA')
        draw.rectangle((x1,y1,x2,y2),outline=color,width=max(2,image.width//250))
        label=f"{d['class_name']} {d['confidence']:.2f}";box=draw.textbbox((0,0),label,font=font)
        w,h=box[2]+10,box[3]+8;x=min(max(x1,0),max(0,image.width-w));y=max(0,y1-h)
        draw.rectangle((x,y,x+w,y+h),fill=color);draw.text((x+5,y),label,font=font,fill='#112337')
    image.save(a.output,quality=94)
    data={'epochs':epoch,'classes':model.names,'detections':detections,'size':list(image.size),'checkpoint_sha256':hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),'backend':'pytorch_cuda' if torch.cuda.is_available() else 'pytorch_cpu','training_started':False}
    Path(a.result).write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')

if __name__=='__main__':main()
