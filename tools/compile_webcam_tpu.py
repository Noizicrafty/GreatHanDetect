"""Compile the existing trained cigarette detector. Calibration only; no training."""
import hashlib,json,os,sys,time,traceback,argparse
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--preset',choices=['O1','O3','O5'],default='O1');a=p.parse_args()
    root=Path(a.root);root.mkdir(parents=True,exist_ok=True);started=time.time()
    def state(stage,**extra):
        record={'stage':stage,'preset':a.preset,'pid':os.getpid(),'at':time.time(),'elapsed_s':time.time()-started,**extra};(root/('compile_'+a.preset+'.status.json')).write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
    try:
        import numpy as np,onnx,torch
        from ultralytics import YOLO
        import tpu_framework as tpu
        torch.set_num_threads(2)
        checkpoint=root/'last.pt';expected='5b9df5d26785a3848643b7603036c2219dea4aedd6a19373dadaaa56182e519d'
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=expected:raise ValueError('Checkpoint hash mismatch')
        source=root/'last.onnx'
        if not source.exists():
            state('export_416_static');model=YOLO(checkpoint)
            model.export(format='onnx',imgsz=416,opset=17,dynamic=False,batch=1,nms=False,simplify=False,device='cpu')
        graph=onnx.load(source);onnx.checker.check_model(graph)
        inputs={x.name:{'shape':[d.dim_value for d in x.type.tensor_type.shape.dim],'dtype':'float32'} for x in graph.graph.input}
        outputs={x.name:{'shape':[d.dim_value for d in x.type.tensor_type.shape.dim],'dtype':'float32'} for x in graph.graph.output}
        if list(inputs.values())[0]['shape']!=[1,3,416,416] or list(outputs.values())[0]['shape']!=[1,8,3549]:raise ValueError('Unexpected graph signature')
        state('condition_onnx',inputs=inputs,outputs=outputs)
        conditioned=tpu.preliminary_model_conditioning(graph);onnx.save(conditioned,root/'trained26.optimized.onnx')
        regular=tpu.RegularModel(model=conditioned);threshold_file=root/'trained26.thresholds.json'
        if threshold_file.exists():thresholds=tpu.RegularModel.load_thresholds(str(threshold_file))
        else:
            data=np.load(root/'calibration_full416.npz');state('calibrate_no_training',samples=int(data['images'].shape[0]))
            thresholds=regular.calibrate_torch(calibration_data={'images':data['images'].astype(np.float32)},device='cpu')
            tpu.RegularModel.dump_thresholds(str(threshold_file),thresholds)
        state('quantize');quantized=regular.quantize(thresholds);quantized.save(file_dir=str(root),file_name='trained26.qm.onnx')
        state('build_network');network,renaming=tpu.Network.from_quantized_model(quantized);network.set_batch(1)
        state('compile_h1m');executable,descriptions=tpu.compiler.compile_(hardware_parameters=tpu.TPU_H1M,network=network,parameters=getattr(tpu,a.preset))
        target=Path('/auto/cache')/os.environ.get('JUPYTERHUB_USER','hackathon61961')/('smoking_trained26_416_'+a.preset.lower()+'.tpu')
        tpu.TpuProgram.from_executable(executable,descriptions).to_file(str(target))
        import shutil;shutil.copy2(target,root/target.name)
        manifest={'status':'compiled_not_runtime_verified','checkpoint_sha256':expected,'epochs':26,'onnx_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'program_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'program':str(target),'inputs':inputs,'outputs':outputs,'classes':['cigarette','person','smoke','vape'],'input_size':416,'target':'TPU_H1M','preset':a.preset,'training_started':False,'calibration_sha256':hashlib.sha256((root/'calibration_full416.npz').read_bytes()).hexdigest(),'duration_s':time.time()-started,'renaming':renaming}
        (root/('manifest_'+a.preset+'.json')).write_text(json.dumps(manifest,indent=2));state('compiled',program=str(target),program_sha256=manifest['program_sha256'])
    except Exception as e:state('failed',error=type(e).__name__,message=str(e));traceback.print_exc();raise

if __name__=='__main__':main()
