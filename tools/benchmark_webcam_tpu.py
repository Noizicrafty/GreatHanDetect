"""Real LinQ runtime evidence for our exact checkpoint; no CPU fallback for TPU calls."""
import argparse,hashlib,json,os,sys,time,threading
from pathlib import Path
import numpy as np,cv2,onnxruntime as ort
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from greathan.backends import AmqpLinQBackend,BackendError
from tools.common import letterbox
from tools.decoder import decode_objects

CLASSES=['cigarette','person','smoke','vape']
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--preset',default='O1');p.add_argument('--requests',type=int,default=30);a=p.parse_args();root=Path(a.root)
    manifest=json.loads((root/('manifest_'+a.preset+'.json')).read_text());spec={**manifest,'remote_program':manifest['program']}
    config={'host':'10.0.114.30','port':5672,'exchange':'tpu.direct','routing_key':'input','timeout_s':15,'username_env':'LINQ_AMQP_USER','password_env':'LINQ_AMQP_PASSWORD'}
    report={'checkpoint_sha256':manifest['checkpoint_sha256'],'program_sha256':hashlib.sha256(Path(manifest['program']).read_bytes()).hexdigest(),'program':manifest['program'],'preset':a.preset,'backend':'linq_amqp','training_started':False,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'quality_scope':'Nine previously recorded random test images. Diagnostic subset, not independent event accuracy.'}
    if report['program_sha256']!=manifest['program_sha256']:raise ValueError('Runtime program differs from compilation')
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    cpu=ort.InferenceSession(str(root/'last.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    backend=AmqpLinQBackend(config);samples=json.loads((root/'samples.json').read_text())['records'];records=[]
    for sample in samples:
        frame=cv2.imread(str(root/'samples'/sample['file']));started=time.perf_counter();tensor,meta=letterbox(frame,416);prep=time.perf_counter()-started
        raw_cpu=cpu.run(None,{'images':tensor})[0];raw_tpu,timing=backend.infer(spec,{'images':tensor});raw=next(iter(raw_tpu.values()))
        def decoded(x):return [{'class_name':n,'confidence':c,'xyxy':[float(v) for v in b]} for n,c,b in decode_objects(x,meta,CLASSES,conf=.25,iou=.7)]
        pred_cpu,pred_tpu=decoded(raw_cpu),decoded(raw)
        record={'sample_id':sample['sample_id'],'file':sample['file'],'input_sha256':hashlib.sha256(tensor.tobytes()).hexdigest(),'output_sha256':hashlib.sha256(raw.tobytes()).hexdigest(),'ground_truth':sample['ground_truth'],'cpu':pred_cpu,'tpu':pred_tpu,'preprocess_s':prep,'timing':timing}
        records.append(record)
        annotated=frame.copy()
        for d in pred_tpu:
            box=[round(x) for x in d['xyxy']];cv2.rectangle(annotated,box[:2],box[2:],(30,190,255),2);cv2.putText(annotated,f"TPU {d['class_name']} {d['confidence']:.2f}",(max(0,box[0]),max(20,box[1]-6)),cv2.FONT_HERSHEY_SIMPLEX,.6,(30,190,255),2)
        cv2.imwrite(str(root/(f"tpu_{a.preset}_sample{sample['sample_id']}.jpg")),annotated)
    report['samples']=records;(root/('runtime_'+a.preset+'.json')).write_text(json.dumps(report,indent=2))
    print('Quality diagnostic completed',flush=True)
    frame=cv2.imread(str(root/'samples'/samples[3]['file']));tensor,meta=letterbox(frame,416)
    sequential=[];started=time.perf_counter()
    for i in range(a.requests):
        t=time.perf_counter();out,timing=backend.infer(spec,{'images':tensor});sequential.append({'elapsed_s':time.perf_counter()-t,'timing':timing})
    elapsed=time.perf_counter()-started
    report['sequential']={'requests':a.requests,'duration_s':elapsed,'fps':a.requests/elapsed,'p50_s':float(np.percentile([x['elapsed_s'] for x in sequential],50)),'p95_s':float(np.percentile([x['elapsed_s'] for x in sequential],95)),'records':sequential}
    backend.close();print('Sequential',report['sequential']['fps'],flush=True)
    report['concurrency']=[]
    for workers in [2,4]:
        def worker(count):
            b=AmqpLinQBackend(config);results=[]
            try:
                for _ in range(count):
                    t=time.perf_counter();out,timing=b.infer(spec,{'images':tensor});results.append({'elapsed_s':time.perf_counter()-t,'timing':timing})
            finally:b.close()
            return results
        n=max(4,a.requests//workers);started=time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:groups=list(pool.map(worker,[n]*workers))
        elapsed=time.perf_counter()-started;results=[x for group in groups for x in group]
        report['concurrency'].append({'inflight':workers,'requests':len(results),'duration_s':elapsed,'fps':len(results)/elapsed,'p50_s':float(np.percentile([x['elapsed_s'] for x in results],50)),'p95_s':float(np.percentile([x['elapsed_s'] for x in results],95)),'records':results})
        print('Inflight',workers,'FPS',len(results)/elapsed,flush=True)
    # A deliberately absent team-scoped program must fail, proving this path has no CPU fallback.
    negative=AmqpLinQBackend(config)
    try:
        negative.infer({**spec,'remote_program':str(Path(manifest['program']).with_name('nonexistent_greathan_evidence.tpu'))},{'images':tensor});report['missing_program_control']='UNEXPECTED_SUCCESS'
    except BackendError as e:report['missing_program_control']={'expected_error':str(e)}
    finally:negative.close()
    (root/('runtime_'+a.preset+'.json')).write_text(json.dumps(report,indent=2));print(json.dumps({'preset':a.preset,'sequential_fps':report['sequential']['fps'],'concurrency':[{k:v for k,v in x.items() if k!='records'} for x in report['concurrency']],'missing_program_control':report['missing_program_control']}),flush=True)

if __name__=='__main__':main()
