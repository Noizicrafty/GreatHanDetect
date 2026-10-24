"""Team Jupyter JPEG -> LinQ -> decoded JSON service, loopback only. No CPU fallback."""
import argparse,base64,hashlib,io,json,os,sys,threading,time,queue
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from greathan.backends import AmqpLinQBackend,BackendError
from tools.common import letterbox
from tools.decoder import decode_objects

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--preset',default='O1');p.add_argument('--port',type=int,default=8094)
    p.add_argument('--bind',choices=['127.0.0.1','0.0.0.0'],default='127.0.0.1')
    p.add_argument('--program',help='Local copy used for SHA-256 verification')
    p.add_argument('--remote-program',default=os.environ.get('LINQ_REMOTE_PROGRAM'),help='Same program on organiser /auto/cache filesystem')
    a=p.parse_args();root=Path(a.root)
    manifest=json.loads((root/('manifest_'+a.preset+'.json')).read_text());spec={**manifest,'remote_program':manifest['program']};slots=queue.Queue()
    if hashlib.sha256(Path(a.program or manifest['program']).read_bytes()).hexdigest()!=manifest['program_sha256']:raise ValueError('TPU artifact hash mismatch')
    spec['remote_program']=a.remote_program or manifest['program']
    AmqpLinQBackend.program_uri(spec['remote_program'])
    config={'host':os.environ.get('LINQ_AMQP_HOST','10.0.114.30'),'port':int(os.environ.get('LINQ_AMQP_PORT','5672')),'exchange':os.environ.get('LINQ_AMQP_EXCHANGE','tpu.direct'),'routing_key':os.environ.get('LINQ_AMQP_ROUTING_KEY','input'),'virtual_host':os.environ.get('LINQ_AMQP_VHOST','/'),'timeout_s':12,'username_env':'LINQ_AMQP_USER','password_env':'LINQ_AMQP_PASSWORD'}
    # A dedicated thread owns pika across all HTTP requests; reconnects are avoided.
    backends=[AmqpLinQBackend(config) for _ in range(2)];executors=[ThreadPoolExecutor(max_workers=1) for _ in range(2)];stopped=threading.Event()
    for i in range(2):slots.put(i)
    def heartbeat(backend):
        try:
            if backend.connection is not None and backend.connection.is_open:backend.connection.process_data_events(time_limit=0)
        except Exception:backend.close()
    def maintain():
        while not stopped.wait(8):
            for backend,executor in zip(backends,executors):executor.submit(heartbeat,backend)
    threading.Thread(target=maintain,daemon=True).start()
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def send(self,value,status=200):
            data=json.dumps(value).encode();self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers()
            try:self.wfile.write(data)
            except ConnectionError:pass
        def do_GET(self):
            return self.send({'ready':True,'backend':'linq_amqp','preset':a.preset,'checkpoint_sha256':manifest['checkpoint_sha256'],'program_sha256':manifest['program_sha256'],'target':manifest['target'],'training_started':False})
        def do_POST(self):
            slot=None;started=time.perf_counter()
            try:
                size=int(self.headers.get('Content-Length',0))
                if not 0<size<=4*1024*1024:return self.send({'error':'Frame too large'},413)
                body=json.loads(self.rfile.read(size));raw=base64.b64decode(body['image'],validate=True)
                with Image.open(io.BytesIO(raw)) as checked:
                    if checked.width*checked.height>2000000:raise ValueError('Resolution too large')
                image=cv2.imdecode(np.frombuffer(raw,dtype=np.uint8),cv2.IMREAD_COLOR)
                if image is None or image.shape[0]*image.shape[1]>2000000:raise ValueError('Invalid image or resolution')
                try:slot=slots.get_nowait()
                except queue.Empty:return self.send({'error':'LinQ занят двумя кадрами; повторите позже'},409)
                tensor,meta=letterbox(image,416);prepared=time.perf_counter()
                outputs,timing=executors[slot].submit(backends[slot].infer,spec,{'images':tensor}).result(timeout=15);decoded=decode_objects(next(iter(outputs.values())),meta,manifest['classes'],conf=.25,iou=.7)
                detections=[{'class_id':manifest['classes'].index(n),'class_name':n,'confidence':c,'xyxy':[float(x) for x in b]} for n,c,b in decoded]
                return self.send({'prediction':{'epochs':26,'checkpoint_sha256':manifest['checkpoint_sha256'],'program_sha256':manifest['program_sha256'],'classes':dict(enumerate(manifest['classes'])),'size':[image.shape[1],image.shape[0]],'detections':detections,'backend':'linq_amqp','preset':a.preset,'training_started':False,'inference_ms':round(timing['total_s']*1000,1),'timing':{**timing,'jpeg_preprocess_s':prepared-started,'server_total_s':time.perf_counter()-started}},'saved':False})
            except (ValueError,KeyError,TypeError):return self.send({'error':'Некорректный кадр'},400)
            except BackendError as e:return self.send({'error':str(e)},503)
            except Exception as e:print(type(e).__name__,str(e),flush=True);return self.send({'error':'LinQ inference failed'},503)
            finally:
                if slot is not None:slots.put(slot)
        def log_message(self,fmt,*args):pass
    server=ThreadingHTTPServer((a.bind,a.port),Handler);server.daemon_threads=True
    print(json.dumps({'listen':f'{a.bind}:{a.port}','backend':'linq_amqp','preset':a.preset,'program_sha256':manifest['program_sha256']}),flush=True)
    try:server.serve_forever()
    finally:
        stopped.set();server.server_close()
        for backend,executor in zip(backends,executors):executor.submit(backend.close).result(timeout=5);executor.shutdown()

if __name__=='__main__':main()
