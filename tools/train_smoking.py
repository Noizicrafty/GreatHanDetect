"""Train a specialised detector on the audited YOLO dataset, then validate and export best weights."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
os.environ.setdefault('MPLBACKEND','Agg')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True)
    parser.add_argument('--out',default='results/training')
    parser.add_argument('--weights',default='yolov8n.pt')
    parser.add_argument('--epochs',type=int,default=40)
    parser.add_argument('--imgsz',type=int,default=640)
    parser.add_argument('--batch',type=int,default=16)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--device',default='0')
    parser.add_argument('--rare-repeat',type=int,default=4)
    parser.add_argument('--patience',type=int,default=8)
    args=parser.parse_args()
    import yaml
    import torch
    from ultralytics import YOLO
    if args.device!='cpu' and not torch.cuda.is_available():raise SystemExit('CUDA unavailable; do not silently train on CPU')
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    state_path=out/'training_status.json'
    started=time.time()
    def state(stage,**values):
        target=state_path.with_suffix('.tmp')
        target.write_text(json.dumps({'stage':stage,'pid':os.getpid(),'started_at':started,'updated_at':time.time(),**values},indent=2),encoding='utf-8')
        target.replace(state_path)
    data_path=Path(args.data).resolve()
    config=yaml.safe_load(data_path.read_text(encoding='utf-8'))
    names=config['names']
    if isinstance(names,dict):names=[names[k] for k in sorted(names)]
    # Oversample rare training classes only. Holdout lists remain byte-for-byte unchanged.
    rare={i for i,name in enumerate(names) if name.lower() in ('smoking','vape')}
    images=Path(config['train']).read_text(encoding='utf-8').splitlines()
    balanced=[];rare_images=0
    for value in images:
        if not value.strip():continue
        image=Path(value)
        label=image.parent.parent/'labels'/(image.stem+'.txt')
        classes={int(line.split()[0]) for line in label.read_text(encoding='utf-8').splitlines() if line.strip()}
        repeat=args.rare_repeat if classes&rare else 1
        rare_images+=bool(classes&rare)
        balanced.extend([str(image)]*repeat)
    random.Random(42).shuffle(balanced)
    train_list=out/'train_balanced.txt';train_list.write_text('\n'.join(balanced)+'\n',encoding='utf-8')
    config['train']=str(train_list)
    training_yaml=out/'data.yaml';training_yaml.write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    metadata={'source_data_sha256':hashlib.sha256(data_path.read_bytes()).hexdigest(),'val_list_sha256':hashlib.sha256(Path(config['val']).read_bytes()).hexdigest(),
              'test_list_sha256':hashlib.sha256(Path(config['test']).read_bytes()).hexdigest(),'rare_classes':[names[i] for i in sorted(rare)],
              'rare_repeat':args.rare_repeat,'rare_training_images':rare_images,'training_items':len(balanced),'unique_training_images':len(set(balanced)),
              'seed':42,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}
    (out/'training_manifest.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    state('training',**metadata)
    model=YOLO(args.weights)
    def on_epoch(trainer):
        state('training',epoch=trainer.epoch+1,epochs=trainer.epochs,metrics={k:float(v) for k,v in trainer.metrics.items()},save_dir=str(trainer.save_dir))
    model.add_callback('on_fit_epoch_end',on_epoch)
    try:
        model.train(data=str(training_yaml),epochs=args.epochs,imgsz=args.imgsz,batch=args.batch,workers=args.workers,
                    device=args.device,seed=42,deterministic=True,patience=args.patience,project=str(out),name='smoking',
                    exist_ok=False,amp=True,cache=False,close_mosaic=5,save_period=5,plots=True)
        best=Path(model.trainer.best)
        trained=YOLO(str(best))
        state('validating_test',best=str(best))
        metrics=trained.val(data=str(training_yaml),split='test',device=args.device,imgsz=args.imgsz,batch=args.batch,plots=True,project=str(out),name='heldout_test')
        summary={'weights':str(best),'weights_sha256':hashlib.sha256(best.read_bytes()).hexdigest(),'split':'test',
                 'metrics':{k:float(v) for k,v in metrics.results_dict.items()},'class_names':trained.names,
                 'class_indices_evaluated':[int(i) for i in metrics.box.ap_class_index],
                 'class_map50':[float(x) for x in metrics.box.ap50],'class_map50_95':[float(x) for x in metrics.box.ap],
                 'limitations':'Object-level heldout image evaluation, not event/video or industrial field validation',**metadata}
        (out/'heldout_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        for size in sorted(set([416,args.imgsz])):
            state('exporting_onnx',size=size)
            exported=Path(trained.export(format='onnx',imgsz=size,opset=17,dynamic=False,batch=1,nms=False,simplify=False,device='cpu'))
            import shutil
            shutil.copy2(exported,out/f'smoking_{size}.onnx')
        state('complete',best=str(best),heldout_summary=str(out/'heldout_summary.json'))
    except Exception as exc:
        state('failed',error=type(exc).__name__,message=str(exc));raise


if __name__=='__main__':main()
