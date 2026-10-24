"""Webcam boundaries and real resident-model checks. Run with the model environment."""
import base64,io,json,tempfile,threading,unittest,urllib.request,urllib.error,sys,time
import os
from pathlib import Path
from types import SimpleNamespace
from http.server import ThreadingHTTPServer
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_presentation_demo import Demo,make_handler,ROOT

class WebcamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.app=Demo(SimpleNamespace(root=str(ROOT/'demo/ff'),data=cls.temp.name,model_python=os.environ.get('DEMO_MODEL_PYTHON',sys.executable),checkpoint=os.environ.get('DEMO_CHECKPOINT',str(ROOT/'results/training_roboflow_v4/smoking/weights/last.pt'))))
        h=make_handler(cls.app);h.log_message=lambda *_:None;cls.server=ThreadingHTTPServer(('127.0.0.1',0),h);cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start();cls.url=f'http://127.0.0.1:{cls.server.server_port}'
    @classmethod
    def tearDownClass(cls):cls.server.shutdown();cls.server.server_close();cls.thread.join();cls.app.close_webcam();cls.temp.cleanup()
    def post(self,body,origin=None):
        req=urllib.request.Request(self.url+'/api/demo/webcam',data=json.dumps(body).encode(),headers={'Content-Type':'application/json',**({'Origin':origin} if origin else {})})
        try:
            with urllib.request.urlopen(req,timeout=80) as r:return r.status,json.load(r)
        except urllib.error.HTTPError as e:return e.code,json.load(e)
    def encoded(self,image):
        out=io.BytesIO();image.save(out,format='JPEG');return base64.b64encode(out.getvalue()).decode()
    def test_01_modules_and_wasm_mime(self):
        for path,kind in [('webcam.mjs','text/javascript'),('webcam-face-worker.mjs','text/javascript'),('assets/mediapipe/wasm/vision_wasm_internal.wasm','application/wasm')]:
            with urllib.request.urlopen(self.url+'/'+path) as r:self.assertEqual(r.headers['Content-Type'],kind);r.read()
    def test_02_bad_images_origin_and_size(self):
        for data in [{'image':'invalid!'}, {'image':123}, {'image':base64.b64encode(b'no image').decode()}, {'image':'a'*(3*1024*1024+1)}]:self.assertEqual(self.post(data)[0],400)
        self.assertEqual(self.post({},'https://untrusted.invalid')[0],403)
        self.assertEqual(self.post({'image':self.encoded(Image.new('RGB',(2000,1100)))})[0],400)
    def test_03_real_inference_keeps_worker_and_does_not_save_frames(self):
        before=[e['id'] for e in self.app.events()];files=list((self.app.data/'uploads').iterdir());timings=[];pid=None
        for _ in range(3):
            started=time.perf_counter();code,data=self.post({'image':base64.b64encode((ROOT/'demo/ff/assets/photos/trained_sample.jpg').read_bytes()).decode()});timings.append(round((time.perf_counter()-started)*1000))
            self.assertEqual(code,200,data);p=data['prediction'];self.assertEqual(p['epochs'],26);self.assertFalse(p['training_started']);self.assertTrue(any(d['class_name']=='cigarette' for d in p['detections']));self.assertFalse(data['saved'])
            if pid is not None:self.assertEqual(self.app.webcam_worker.pid,pid)
            pid=self.app.webcam_worker.pid
        self.assertEqual(before,[e['id'] for e in self.app.events()]);self.assertEqual(files,list((self.app.data/'uploads').iterdir()))
        out=ROOT/'results/webcam_qa';out.mkdir(exist_ok=True);(out/'resident_model.json').write_text(json.dumps({'request_ms':timings,'worker_pid':pid,'checkpoint_sha256':p['checkpoint_sha256'],'epochs':p['epochs'],'saved_frames':0},indent=2))
    def test_04_negative_and_busy(self):
        body={'image':self.encoded(Image.new('RGB',(320,240),(10,15,20)))}
        code,data=self.post(body);self.assertEqual(code,200);self.assertEqual(data['prediction']['detections'],[])
        self.app.predict_lock.acquire()
        try:self.assertEqual(self.post(body)[0],409)
        finally:self.app.predict_lock.release()
    def test_05_worker_restarts_after_exit(self):
        old=self.app.webcam_worker.pid;self.app.webcam_worker.kill();self.app.webcam_worker.wait()
        self.assertEqual(self.post({'image':self.encoded(Image.new('RGB',(320,240)))})[0],200)
        self.assertNotEqual(old,self.app.webcam_worker.pid)

if __name__=='__main__':unittest.main(verbosity=2)
