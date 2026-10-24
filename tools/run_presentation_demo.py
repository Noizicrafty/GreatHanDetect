"""Presentation demo; loopback by default, explicit HTTPS origin behind a deployment proxy."""
from pathlib import Path
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from urllib.parse import urlsplit,parse_qs,unquote
from datetime import datetime,timezone,timedelta
from concurrent.futures import ThreadPoolExecutor,TimeoutError as FutureTimeout
from contextlib import contextmanager
import argparse,base64,io,json,sqlite3,subprocess,sys,threading,uuid,zipfile,html,hashlib,os
from urllib.request import Request,build_opener,ProxyHandler
from urllib.error import HTTPError,URLError

ROOT=Path(__file__).resolve().parents[1]
SEED=[
 {'id':'EVT-2046','category':'helmet','title':'Отсутствие каски','camera':'CAM-03','zone':'Цех № 2 · операторская зона','time':'20:02:16','confidence':.92,'severity':'critical','status':'new','image':'assets/photos/missing_helmet.png','active':True},
 {'id':'EVT-2050','category':'harness','title':'Отсутствие страховочного троса','camera':'CAM-04','zone':'Участок высотных работ','time':'20:05:31','confidence':.87,'severity':'critical','status':'auto','image':'assets/photos/missing_safety_hook.png','active':True},
 {'id':'EVT-2048','category':'smoking','title':'Курение','camera':'CAM-01','zone':'Проход между участками','time':'20:07:14','confidence':.72,'severity':'medium','status':'new','image':'assets/photos/smoking.png','active':True},
 {'id':'EVT-2049','category':'running','title':'Бег по рабочей зоне','camera':'CAM-08','zone':'Производственная территория','time':'20:09:32','confidence':.81,'severity':'low','status':'auto','image':'assets/photos/running_workplace_still.png','active':True},
 {'id':'EVT-2045','category':'helmet','title':'Отсутствие каски','camera':'CAM-03','zone':'Цех № 2 · операторская зона','time':'19:58:43','confidence':.94,'severity':'critical','status':'confirmed','image':'assets/photos/missing_helmet.png','active':False},
]
STATUSES={'new':'На проверке','confirmed':'Подтверждено','rejected':'Отклонено','auto':'Автоматически зарегистрировано'}

class Demo:
    def __init__(self,args):
        self.args=args;self.web=Path(args.root).resolve();self.data=Path(args.data).resolve()
        self.data.mkdir(parents=True,exist_ok=True);(self.data/'uploads').mkdir(exist_ok=True)
        self.db=self.data/'demo.sqlite3';self.predict_lock=threading.Lock()
        self.webcam_worker=None;self.webcam_reader=None
        with self.connect() as db:db.execute('CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY, body TEXT NOT NULL)')
        if not self.events():self.reset()
    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.db,timeout=10)
        try:
            db.execute('PRAGMA journal_mode=WAL');yield db;db.commit()
        except Exception:
            db.rollback();raise
        finally:db.close()
    def reset(self):
        with self.connect() as db:
            for item in SEED:
                e={**item,'source':'presentation_demo','date':'23.10.2026'}
                db.execute('INSERT INTO events VALUES(?,?) ON CONFLICT(id) DO UPDATE SET body=excluded.body',(e['id'],json.dumps(e,ensure_ascii=False)))
    def events(self):
        with self.connect() as db:return [json.loads(r[0]) for r in db.execute("SELECT body FROM events ORDER BY CASE WHEN id LIKE 'FRAME-%' THEN 0 ELSE 1 END, CASE WHEN id LIKE 'FRAME-%' THEN -rowid ELSE rowid END")]
    def state(self):
        es=self.events();return {'events':es,'pending':sum(e['status']=='new' for e in es),'demo':True,'reference_date':'23.10.2026','model_ready':Path(self.args.model_python).is_file() and Path(self.args.checkpoint).is_file(),'model_name':'YOLOv8n · checkpoint 26 эпох','training_enabled':False,'public_origin':getattr(self.args,'public_origin',None)}
    def review(self,id,status):
        if status not in ('confirmed','rejected'):raise ValueError('Недопустимое решение')
        with self.connect() as db:
            row=db.execute('SELECT body FROM events WHERE id=?',(id,)).fetchone()
            if not row:raise KeyError('Событие не найдено')
            event=json.loads(row[0]);event.update(status=status,reviewed_at=datetime.now(timezone.utc).isoformat())
            db.execute('UPDATE events SET body=? WHERE id=?',(json.dumps(event,ensure_ascii=False),id))
        return event
    def predict(self,body):
        if not self.state()['model_ready']:raise ValueError('Не заданы Python с Ultralytics и существующий checkpoint')
        if not self.predict_lock.acquire(blocking=False):raise RuntimeError('Предыдущий кадр ещё обрабатывается')
        try:
            from PIL import Image
            raw=base64.b64decode(body.get('image',''),validate=True)
            if len(raw)>10*1024*1024:raise ValueError('Размер изображения больше 10 МБ')
            try:
                image=Image.open(io.BytesIO(raw))
                if image.width*image.height>24000000:raise ValueError('Слишком большое разрешение изображения')
                image.verify();image=Image.open(io.BytesIO(raw)).convert('RGB')
            except (OSError,Image.DecompressionBombError) as e:raise ValueError('Файл не является корректным изображением') from e
            normalized=io.BytesIO();image.save(normalized,format='JPEG',quality=95)
            digest=hashlib.sha256(normalized.getvalue());digest.update(Path(self.args.checkpoint).read_bytes())
            id='FRAME-'+digest.hexdigest()[:16];folder=self.data/'uploads'
            source=folder/(id+'.input.jpg');annotated=folder/(id+'.jpg');result=folder/(id+'.json')
            with self.connect() as db:existing=db.execute('SELECT body FROM events WHERE id=?',(id,)).fetchone()
            if existing and annotated.exists():
                event=json.loads(existing[0])
                return {'prediction':event['prediction'],'image':event['image'],'event':event,'duplicate':True,'message':'Этот кадр уже есть в журнале. Повторное событие не создано; решение оператора сохранено.'}
            source.write_bytes(normalized.getvalue())
            subprocess.run([self.args.model_python,str(ROOT/'tools/presentation_predict.py'),'--checkpoint',self.args.checkpoint,'--image',str(source),'--output',str(annotated),'--result',str(result)],timeout=90,check=True,capture_output=True)
            prediction=json.loads(result.read_text(encoding='utf-8'))
            objects=[d for d in prediction['detections'] if d['class_name'] in ('cigarette','smoke','vape')]
            event=None
            if objects:
                event={'id':id,'category':'smoking','title':'Признаки курения · кадр','camera':'WEBCAM-01' if body.get('source')=='webcam' else 'LOCAL-01','zone':'Веб-камера · сохранённый кадр' if body.get('source')=='webcam' else 'Загруженное изображение','date':datetime.now(timezone(timedelta(hours=3))).strftime('%d.%m.%Y'),'time':datetime.now(timezone(timedelta(hours=3))).strftime('%H:%M:%S'),'confidence':max(d['confidence'] for d in objects),'severity':'medium','status':'new','image':'/demo-uploads/'+annotated.name,'input_image':str(source),'prediction':prediction,'active':True,'source':'trained_checkpoint','note':'Объектная детекция кадра; временной эпизод не подтверждён.'}
                with self.connect() as db:db.execute('INSERT INTO events VALUES(?,?)',(id,json.dumps(event,ensure_ascii=False)))
            return {'prediction':prediction,'image':'/demo-uploads/'+annotated.name,'event':event,'message':'Объекты обнаружены; событие отправлено оператору.' if event else 'Сигарета, дым или вейп при пороге 0,25 не обнаружены. Событие не создано.'}
        finally:self.predict_lock.release()
    def select(self,query):
        es=self.events();category=query.get('category',['all'])[0];search=query.get('q',[''])[0].lower()
        return [e for e in es if (category=='all' or e['category']==category) and (not search or search in ' '.join(str(v) for v in e.values()).lower())]
    def close_webcam(self):
        worker=self.webcam_worker;self.webcam_worker=None
        if worker:
            if worker.poll() is None:worker.kill()
            worker.wait(timeout=5)
            worker.stdin.close();worker.stdout.close()
        if self.webcam_reader:self.webcam_reader.shutdown(wait=True,cancel_futures=True);self.webcam_reader=None
    def webcam(self,body):
        """Transient inference: no journal entry and no image files per frame."""
        from PIL import Image
        if not self.state()['model_ready']:raise ValueError('Модель недоступна')
        encoded=body.get('image','')
        if not isinstance(encoded,str) or len(encoded)>3*1024*1024:raise ValueError('Кадр веб-камеры слишком большой')
        try:
            raw=base64.b64decode(encoded,validate=True);image=Image.open(io.BytesIO(raw))
            if image.width*image.height>2000000:raise ValueError('Максимум 2 мегапикселя на кадр')
            image.verify();image=Image.open(io.BytesIO(raw)).convert('RGB')
        except (OSError,Image.DecompressionBombError) as e:raise ValueError('Некорректный кадр веб-камеры') from e
        if not self.predict_lock.acquire(blocking=False):raise RuntimeError('Модель занята другим кадром; повторите позже')
        try:
            image.thumbnail((1280,960));out=io.BytesIO();image.save(out,format='JPEG',quality=90)
            if self.webcam_worker is None or self.webcam_worker.poll() is not None:
                self.close_webcam()
                self.webcam_reader=ThreadPoolExecutor(max_workers=1)
                self.webcam_worker=subprocess.Popen([self.args.model_python,str(ROOT/'tools/presentation_predict.py'),'--checkpoint',self.args.checkpoint,'--stream'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=None,text=True,encoding='utf-8',bufsize=1)
            self.webcam_worker.stdin.write(json.dumps({'image':base64.b64encode(out.getvalue()).decode()})+'\n');self.webcam_worker.stdin.flush()
            line=self.webcam_reader.submit(self.webcam_worker.stdout.readline).result(timeout=60)
            if not line:raise RuntimeError('Процесс модели остановился; повторите запуск')
            prediction=json.loads(line)
            if prediction.get('error'):raise ValueError(prediction['error'])
            return {'prediction':prediction,'saved':False}
        except FutureTimeout:
            self.close_webcam();raise subprocess.TimeoutExpired('webcam inference',60)
        except (BrokenPipeError,OSError,json.JSONDecodeError):
            self.close_webcam();raise RuntimeError('Процесс модели перезапущен; повторите запрос')
        finally:self.predict_lock.release()
    def xlsx(self,query):
        from openpyxl import Workbook
        from openpyxl.styles import Font,PatternFill,Alignment
        wb=Workbook();ws=wb.active;ws.title='События'
        ws.append(['Время','Камера','Место','Событие','Confidence','Статус','Источник','ID'])
        for e in self.select(query):ws.append([e['date']+' '+e['time'],e['camera'],e['zone'],e['title'],e['confidence'],STATUSES[e['status']],'DEMO презентации' if e['source']=='presentation_demo' else 'PyTorch checkpoint 26 эпох',e['id']])
        for c in ws[1]:c.font=Font(color='FFFFFF',bold=True);c.fill=PatternFill('solid',fgColor='1266E8')
        for column,width in zip('ABCDEFGH',[25,14,40,35,16,36,34,25]):ws.column_dimensions[column].width=width
        for row in ws.iter_rows(min_row=2):row[4].number_format='0%'
        ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
        note=wb.create_sheet('Примечания');note.append(['Демонстрационный журнал. Confidence не является метрикой точности.']);note.append(['Кадровая детекция не подтверждает временной эпизод курения.'])
        out=io.BytesIO();wb.save(out);return out.getvalue()
    def pdf(self):
        from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle
        from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        for name,paths in [('DemoFont',['C:/Windows/Fonts/segoeui.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']),('DemoBold',['C:/Windows/Fonts/segoeuib.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'])]:
            if name not in pdfmetrics.getRegisteredFontNames():pdfmetrics.registerFont(TTFont(name,next(p for p in paths if Path(p).exists())))
        es=self.events();styles=getSampleStyleSheet()
        for s in styles.byName.values():s.fontName='DemoFont'
        styles['Title'].fontName='DemoBold';styles['Title'].fontSize=24;styles['Title'].leading=30;styles['Title'].alignment=0
        styles['BodyText'].fontSize=10;styles['BodyText'].leading=15
        cell=ParagraphStyle('cell',fontName='DemoFont',fontSize=8,leading=11)
        out=io.BytesIO();doc=SimpleDocTemplate(out,pagesize=A4,rightMargin=38,leftMargin=38,topMargin=38,bottomMargin=40,title='LinQ Safety - локальный демонстрационный отчёт')
        story=[Paragraph('LinQ Safety Control',styles['Heading2']),Paragraph('Отчёт по инцидентам',styles['Title']),Paragraph('Локальное демо. Материалы презентации и результаты проверки загруженных кадров.',styles['BodyText']),Spacer(1,18)]
        counts=[len(es),sum(e['status']=='confirmed' for e in es),sum(e['status']=='rejected' for e in es),sum(e['status']=='new' for e in es),sum(e['status']=='auto' for e in es)]
        summary=Table([['Всего','Подтверждено','Отклонено','На проверке','Авто'],list(map(str,counts))],colWidths=[103]*5,rowHeights=[32,54])
        summary.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'DemoFont'),('FONTNAME',(0,1),(-1,1),'DemoBold'),('FONTSIZE',(0,1),(-1,1),22),('LEADING',(0,1),(-1,1),28),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TEXTCOLOR',(0,0),(-1,-1),colors.HexColor('#182A43')),('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#EDF4FF')),('TOPPADDING',(0,0),(-1,-1),12),('BOTTOMPADDING',(0,0),(-1,-1),12)]))
        story += [summary,Spacer(1,18),Paragraph('Реестр событий',styles['Heading2'])]
        rows=[[Paragraph(x,cell) for x in ['Камера / время','Событие','Решение','Происхождение']]]
        for e in es:rows.append([Paragraph(html.escape(v),cell) for v in [e['camera']+' / '+e['time'],e['title'],STATUSES[e['status']],'DEMO' if e['source']=='presentation_demo' else 'Checkpoint 26 эпох']])
        table=Table(rows,colWidths=[105,190,105,116],repeatRows=1)
        table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#EAF2FF')),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#DDE5F1')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),9),('BOTTOMPADDING',(0,0),(-1,-1),9)]))
        story += [table,Spacer(1,18),Paragraph('Границы демонстрации',styles['Heading2']),Paragraph('Камеры и нарушения из презентации - демонстрационные материалы, не результаты подключённых моделей. Загруженный кадр анализируется сохранённым YOLOv8n checkpoint после 26 эпох через локальный PyTorch. Это обнаружение объектов, не подтверждение временного эпизода. Confidence не является точностью всей модели. LinQ в этом демо не вызывается; обучение не запускается.',styles['BodyText'])]
        def footer(c,d):c.setFont('DemoFont',8);c.drawString(38,22,'Сформировано '+datetime.now(timezone(timedelta(hours=3))).strftime('%d.%m.%Y %H:%M')+' МСК');c.drawRightString(A4[0]-38,22,str(d.page))
        doc.build(story,onFirstPage=footer,onLaterPages=footer);return out.getvalue()
    def dataset(self):
        out=io.BytesIO()
        with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
            accepted=[e for e in self.events() if e['source']=='trained_checkpoint' and e['status']=='confirmed']
            z.writestr('README_RU.txt','Только подтверждённые оператором кадры реального checkpoint. Боксы - автоматические псевдометки, не исправленная человеком GT-разметка. Исходные DEMO-события исключены.\n')
            z.writestr('data.yaml','names: [cigarette, person, smoke, vape]\npath: .\ntrain: images\n')
            for e in accepted:
                z.write(e['input_image'],'images/'+e['id']+'.jpg');w,h=e['prediction']['size'];labels=[]
                for d in e['prediction']['detections']:
                    x1,y1,x2,y2=d['xyxy'];labels.append(f"{d['class_id']} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
                z.writestr('labels/'+e['id']+'.txt','\n'.join(labels)+'\n')
            z.writestr('manifest.json',json.dumps({'pseudo_labels':True,'frames':len(accepted),'event_ids':[e['id'] for e in accepted]},ensure_ascii=False,indent=2))
        return out.getvalue()

def make_handler(app):
    class Handler(SimpleHTTPRequestHandler):
        extensions_map={**SimpleHTTPRequestHandler.extensions_map,'.mjs':'text/javascript','.js':'text/javascript','.wasm':'application/wasm'}
        def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(app.web),**kwargs)
        def valid(self):
            allowed={'127.0.0.1','localhost'}
            public_origin=getattr(app.args,'public_origin',None)
            if public_origin:allowed.add(urlsplit(public_origin).hostname)
            return self.headers.get('Host','').split(':',1)[0].lower() in allowed
        def send(self,data,kind='application/json; charset=utf-8',status=200,filename=None):
            if isinstance(data,(dict,list)):data=json.dumps(data,ensure_ascii=False).encode()
            self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            if filename:self.send_header('Content-Disposition','attachment; filename="'+filename+'"')
            try:self.end_headers();self.wfile.write(data)
            except ConnectionError:pass # A stopped camera may abort an in-flight HTTP request.
        def do_GET(self):
            if not self.valid():return self.send({'error':'Invalid host'},status=403)
            parsed=urlsplit(self.path);path=unquote(parsed.path)
            try:
                if path=='/':self.send_response(302);self.send_header('Location','/03_cameras.html');self.end_headers();return
                if path=='/favicon.ico':return self.send(b'','image/x-icon',status=204)
                if path=='/api/demo/state':return self.send(app.state())
                if path=='/api/demo/linq-health':return self.linq('/api/demo/linq-health')
                if path=='/api/demo/export.xlsx':return self.send(app.xlsx(parse_qs(parsed.query)),'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',filename='LinQ_Safety_events.xlsx')
                if path=='/api/demo/report.pdf':return self.send(app.pdf(),'application/pdf',filename='LinQ_Safety_report.pdf' if 'download' in parsed.query else None)
                if path=='/api/demo/dataset.zip':return self.send(app.dataset(),'application/zip',filename='LinQ_reviewed_pseudo_labels.zip')
                if path.startswith('/demo-uploads/'):
                    name=Path(path).name
                    if not name.endswith('.jpg') or '/' in path[len('/demo-uploads/'):]:return self.send({'error':'Not found'},status=404)
                    return self.send((app.data/'uploads'/name).read_bytes(),'image/jpeg')
                target=(app.web/path.lstrip('/')).resolve()
                if not target.is_relative_to(app.web) or not target.is_file():return self.send({'error':'Not found'},status=404)
                return super().do_GET()
            except ConnectionError:return
            except FileNotFoundError:return self.send({'error':'Not found'},status=404)
            except Exception as e:return self.send({'error':str(e)},status=500)
        def do_POST(self):
            if not self.valid():return self.send({'error':'Invalid host'},status=403)
            origin=self.headers.get('Origin')
            expected_origin=getattr(app.args,'public_origin',None) or f'http://{self.headers.get("Host")}'
            if origin and origin!=expected_origin:return self.send({'error':'Invalid origin'},status=403)
            try:
                size=int(self.headers.get('Content-Length',0))
                if not 0<=size<=15*1024*1024:return self.send({'error':'Request too large'},status=413)
                body=json.loads(self.rfile.read(size) or b'{}');path=urlsplit(self.path).path
                if not isinstance(body,dict):raise ValueError('Ожидается JSON-объект')
                if path=='/api/demo/webcam/linq':return self.linq('/api/demo/webcam/linq',body)
                if path=='/api/demo/review':return self.send(app.review(body.get('id'),body.get('status')))
                if path=='/api/demo/reset':app.reset();return self.send(app.state())
                if path=='/api/demo/predict':return self.send(app.predict(body))
                if path=='/api/demo/webcam':return self.send(app.webcam(body))
                return self.send({'error':'Not found'},status=404)
            except KeyError as e:return self.send({'error':str(e)},status=404)
            except (ValueError,TypeError) as e:return self.send({'error':str(e)},status=400)
            except RuntimeError as e:return self.send({'error':str(e)},status=409)
            except subprocess.TimeoutExpired:return self.send({'error':'Превышено время инференса'},status=504)
            except Exception:return self.send({'error':'Не удалось обработать запрос. Проверьте окружение модели.'},status=500)
        def linq(self,path,body=None):
            upstream=getattr(app.args,'linq_upstream',None)
            if not upstream:return self.send({'ready':False,'error':'LinQ не подключён. См. docs/REVIEWER_RU.md.'},status=503)
            payload=json.dumps(body).encode() if body is not None else None
            if payload and len(payload)>4*1024*1024:return self.send({'error':'Frame too large'},status=413)
            request=Request(upstream.rstrip('/')+path,data=payload,headers={'Content-Type':'application/json'})
            try:
                with build_opener(ProxyHandler({})).open(request,timeout=25 if body else 4) as response:
                    data=response.read(2*1024*1024+1)
                    if len(data)>2*1024*1024:raise ValueError('Upstream response too large')
                    value=json.loads(data)
                    return self.send(value,status=response.status)
            except HTTPError as e:
                return self.send({'ready':False,'error':'LinQ недоступен или занят'},status=e.code if e.code in (400,409,413,429,503,504) else 502)
            except (URLError,TimeoutError,OSError,ValueError):
                return self.send({'ready':False,'error':'Сервис LinQ недоступен'},status=503)
        def log_message(self,fmt,*args):print(fmt%args,flush=True)
    return Handler

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default=str(ROOT/'demo/ff'));p.add_argument('--data',default=str(ROOT/'results/presentation_demo'))
    default_model_python=next((p for p in [ROOT/'.venv-model/Scripts/python.exe',ROOT/'.venv-model/bin/python',ROOT.parent/'train_env/Scripts/python.exe'] if p.is_file()),ROOT/'.venv-model/Scripts/python.exe')
    p.add_argument('--port',type=int,default=8090);p.add_argument('--model-python',default=str(default_model_python))
    p.add_argument('--bind',choices=['127.0.0.1','0.0.0.0'],default='127.0.0.1')
    p.add_argument('--public-origin',default=os.environ.get('PUBLIC_ORIGIN') or None,help='Exact HTTPS origin accepted behind the deployment reverse proxy')
    p.add_argument('--linq-upstream',default=os.environ.get('LINQ_UPSTREAM_URL') or None,help='Fixed internal JPEG/LinQ HTTP service URL; no CPU fallback')
    p.add_argument('--checkpoint',default=str(ROOT/'results/training_roboflow_v4/smoking/weights/last.pt'))
    args=p.parse_args()
    if args.linq_upstream:
        upstream=urlsplit(args.linq_upstream)
        if upstream.scheme not in ('http','https') or not upstream.hostname or upstream.username or upstream.query or upstream.fragment or upstream.path not in ('','/'):p.error('--linq-upstream must be an HTTP(S) origin without credentials or a path')
    if args.public_origin:
        parsed=urlsplit(args.public_origin)
        if parsed.scheme!='https' or not parsed.hostname or parsed.path or parsed.query or parsed.fragment or parsed.username:p.error('--public-origin must be an HTTPS origin without a path or credentials')
    app=Demo(args)
    if not (app.web/'03_cameras.html').is_file():p.error('Распакуйте final_frames.rar в demo/ (ожидается demo/ff/)')
    server=ThreadingHTTPServer((args.bind,args.port),make_handler(app))
    print(f'LinQ presentation demo: {args.public_origin or "http://127.0.0.1:"+str(server.server_port)}/',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close();app.close_webcam()

if __name__=='__main__':main()
