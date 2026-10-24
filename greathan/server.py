"""Local operator console. Bind to loopback; no external service is required for the UI."""
from __future__ import annotations
import argparse
import csv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import re
import threading
from urllib.parse import parse_qs, urlsplit

from .config import ROOT, load_config, public_config, save_config, validate_config
from .pipeline import Pipeline
from .store import EventStore
from .instance_lock import InstanceLock


class Application:
    def __init__(self, config, data_dir, *, config_path=None, vision_factory=None):
        self.config = config
        self.data_dir = Path(data_dir).resolve()
        self.config_path = Path(config_path or self.data_dir/"runtime_config.json")
        self.instance_lock=InstanceLock(self.data_dir)
        self.store = EventStore(self.data_dir/"journal.sqlite3")
        self.pipeline = Pipeline(self.store,self.data_dir,**({"vision_factory":vision_factory} if vision_factory else {}))
        self.lock = threading.RLock()

    def close(self):
        self.pipeline.stop(wait=True)
        self.instance_lock.close()

    def set_config(self, config):
        config = validate_config(config)
        if self.pipeline.thread and self.pipeline.thread.is_alive():
            raise ValueError("Остановите обработку перед изменением настроек")
        save_config(self.config_path,config)
        self.config = config


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        server_version = "GreatHanDetect/0.2"

        def log_message(self, fmt, *args):
            # URLs may include private source names or query parameters.
            pass

        def send(self, status, data, content_type="application/json; charset=utf-8"):
            body = json.dumps(data,ensure_ascii=False).encode("utf-8") if isinstance(data,(dict,list)) else data
            self.send_response(status)
            self.send_header("Content-Type",content_type)
            self.send_header("Content-Length",str(len(body)))
            self.send_header("Cache-Control","no-store")
            self.send_header("X-Content-Type-Options","nosniff")
            self.send_header("Content-Security-Policy","default-src 'self'; img-src 'self' blob: data:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError):
                pass

        def do_GET(self):
            if not self.valid_host():
                return self.send(403,{"error":"Invalid Host"})
            try:
                self.get()
            except (ValueError,KeyError,TypeError) as exc:
                self.send(400,{"error":str(exc)})
            except Exception:
                self.send(500,{"error":"Internal error; inspect local diagnostics"})

        def valid_host(self):
            port=self.server.server_port
            return self.headers.get("Host","").lower() in (f"127.0.0.1:{port}",f"localhost:{port}")

        def get(self):
            parsed = urlsplit(self.path)
            query = {k:v[-1] for k,v in parse_qs(parsed.query).items()}
            path = parsed.path
            if path in ("/","/app.js","/style.css"):
                name = {"/":"index.html","/app.js":"app.js","/style.css":"style.css"}[path]
                content_type = {"/":"text/html; charset=utf-8","/app.js":"text/javascript; charset=utf-8","/style.css":"text/css; charset=utf-8"}[path]
                return self.send(200,(Path(__file__).parent/"web"/name).read_bytes(),content_type)
            if path == "/api/status":
                return self.send(200,app.pipeline.snapshot())
            if path == "/api/config":
                return self.send(200,public_config(app.config))
            if path == "/api/capabilities":
                from .capabilities import catalogue
                return self.send(200,catalogue())
            if path == "/api/events":
                return self.send(200,app.store.events(query,query.get("limit",100),query.get("offset",0)))
            if path == "/api/notifications":
                if query.get("after") == "latest":
                    latest = app.store.notifications(0)["latest"]
                    return self.send(200,{"items":[],"cursor":latest,"latest":latest})
                return self.send(200,app.store.notifications(int(query.get("after",0))))
            if path == "/api/diagnostics":
                return self.send(200,app.store.diagnostics())
            if path == "/api/runs":
                return self.send(200,app.store.runs())
            if path == "/api/delivery-metrics":
                return self.send(200,app.store.delivery_metrics(query.get("run_id")))
            if path == "/api/export.csv":
                stream = io.StringIO(newline="")
                writer = None
                offset = 0
                while True:
                    page = app.store.events(query,1000,offset)
                    if not page["items"]:
                        break
                    if writer is None:
                        writer = csv.DictWriter(stream,fieldnames=list(page["items"][0]))
                        writer.writeheader()
                    for item in page["items"]:
                        # Spreadsheet formula injection must not execute when opening the export.
                        writer.writerow({k:("'"+v if isinstance(v,str) and v[:1] in ("=","+","-","@") else v) for k,v in item.items()})
                    offset += len(page["items"])
                    if offset >= page["total"]:
                        break
                return self.send(200,("\ufeff"+stream.getvalue()).encode("utf-8"),"text/csv; charset=utf-8")
            if path == "/api/frame.jpg":
                with app.pipeline.lock:
                    jpeg = app.pipeline.jpeg
                if jpeg:
                    return self.send(200,jpeg,"image/jpeg")
                return self.send(404,{"error":"No processed frame yet"})
            if path == "/api/source-preview.jpg":
                import os
                import cv2
                camera = app.config["camera"]
                source = os.environ.get(camera["source_env"],"") if camera.get("source_env") else camera["source"]
                if "://" in source or source.isdecimal():
                    raise ValueError("Preview requires a local video file; start processing for live sources")
                capture = cv2.VideoCapture(source)
                try:
                    ok,frame = capture.read()
                    if not ok:
                        raise ValueError("Cannot preview configured video file")
                    ok,encoded = cv2.imencode(".jpg",frame)
                    if not ok:
                        raise ValueError("Cannot encode preview")
                    return self.send(200,encoded.tobytes(),"image/jpeg")
                finally:
                    capture.release()
            if path.startswith("/api/events/"):
                event_id = path.removeprefix("/api/events/")
                event = app.store.get(event_id)
                return self.send(200,event) if event else self.send(404,{"error":"Event not found"})
            if re.fullmatch(r"/evidence/[a-f0-9]{32}\.jpg",path):
                evidence = app.pipeline.evidence/Path(path).name
                if evidence.is_file():
                    return self.send(200,evidence.read_bytes(),"image/jpeg")
            return self.send(404,{"error":"Not found"})

        def do_POST(self):
            if not self.valid_host():
                return self.send(403,{"error":"Invalid Host"})
            try:
                # Block browser cross-origin writes to the local unauthenticated console.
                origin = self.headers.get("Origin")
                expected = "http://"+self.headers.get("Host","")
                if origin and origin != expected:
                    return self.send(403,{"error":"Cross-origin request rejected"})
                if self.headers.get("Content-Type","").split(";")[0] != "application/json":
                    return self.send(415,{"error":"Use application/json"})
                length = int(self.headers.get("Content-Length",0))
                if not 0 < length <= 256000:
                    return self.send(413,{"error":"Invalid request size"})
                body = json.loads(self.rfile.read(length))
                path = urlsplit(self.path).path
                with app.lock:
                    if path == "/api/config":
                        app.set_config(body)
                        return self.send(200,{"ok":True})
                    if path == "/api/start":
                        return self.send(200,{"run_id":app.pipeline.start(app.config)})
                    if path == "/api/stop":
                        app.pipeline.stop()
                        return self.send(200,{"ok":True})
                    if re.fullmatch(r"/api/events/[a-f0-9]{32}/displayed",path):
                        ok=app.store.displayed(path.split("/")[3])
                        return self.send(200,{"ok":True}) if ok else self.send(404,{"error":"Event not found"})
                    if re.fullmatch(r"/api/events/[a-f0-9]{32}/review",path):
                        event_id = path.split("/")[3]
                        ok = app.store.review(event_id,body["status"],body.get("note",""))
                        return self.send(200,{"ok":True}) if ok else self.send(404,{"error":"Event not found"})
                self.send(404,{"error":"Not found"})
            except (ValueError,KeyError,TypeError) as exc:
                self.send(400,{"error":str(exc)})
            except Exception:
                self.send(500,{"error":"Internal error; inspect diagnostics"})
    return Handler


def main():
    parser = argparse.ArgumentParser(description="GreatHanDetect operator console")
    parser.add_argument("--config",default="configs/application.yaml")
    parser.add_argument("--data",default="results/application")
    parser.add_argument("--port",type=int,default=8080)
    args = parser.parse_args()
    runtime_path = Path(args.data)/"runtime_config.json"
    config = load_config(runtime_path if runtime_path.exists() else args.config)
    app = Application(config,args.data,config_path=runtime_path)
    server = ThreadingHTTPServer(("127.0.0.1",args.port),make_handler(app))
    print(f"GreatHanDetect: http://127.0.0.1:{server.server_port} | backend={config['backend']} | scenario={config['scenario']}",flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.close()
        server.server_close()


if __name__ == "__main__":
    main()
