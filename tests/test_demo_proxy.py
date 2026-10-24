"""Compose bridge boundaries; deterministic HTTP stubs, no TPU or weights required."""
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from tools.run_presentation_demo import Demo, ROOT, make_handler


class DemoProxyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.args = SimpleNamespace(root=ROOT/'demo/ff', data=self.temporary.name,
                                    checkpoint='/missing.pt', model_python='/missing', linq_upstream=None)
        self.app = Demo(self.args)
        self.seen = []
        owner = self

        class Upstream(BaseHTTPRequestHandler):
            def do_GET(self):
                owner.seen.append(self.path)
                self.send_response(200); self.end_headers()
                self.wfile.write(b'{"ready":true,"backend":"linq_amqp"}')

            def do_POST(self):
                owner.seen.append((self.path, json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
                self.send_response(409); self.end_headers(); self.wfile.write(b'{}')

            def log_message(self, *_): pass

        self.upstream = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(self.app))
        self.server.RequestHandlerClass.log_message = lambda *_: None
        for server in (self.upstream, self.server):
            threading.Thread(target=server.serve_forever, daemon=True).start()
        self.base = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        for server in (self.server, self.upstream):
            server.shutdown(); server.server_close()
        self.app.close_webcam(); self.temporary.cleanup()

    def request(self, path, body=None, origin=None):
        request = urllib.request.Request(self.base+path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={'Content-Type':'application/json', **({'Origin':origin} if origin else {})})
        try:
            with urllib.request.urlopen(request, timeout=5) as r: return r.status, json.load(r)
        except urllib.error.HTTPError as e: return e.code, json.load(e)

    def connect_upstream(self):
        self.args.linq_upstream = f'http://127.0.0.1:{self.upstream.server_port}'

    def test_absent_tpu_is_explicit_503(self):
        self.assertEqual(self.request('/api/demo/linq-health')[0], 503)
        self.assertEqual(self.request('/api/demo/webcam/linq', {'image':'a'})[0], 503)
        self.assertEqual(self.seen, [])

    def test_health_is_forwarded(self):
        self.connect_upstream()
        status, body = self.request('/api/demo/linq-health')
        self.assertEqual(status, 200); self.assertEqual(body['backend'], 'linq_amqp')
        self.assertEqual(self.seen, ['/api/demo/linq-health'])

    def test_busy_stays_busy_without_cpu_fallback_or_journal_change(self):
        self.connect_upstream(); before = self.app.events()
        self.assertEqual(self.request('/api/demo/webcam/linq', {'image':'test'})[0], 409)
        self.assertEqual(self.seen, [('/api/demo/webcam/linq', {'image':'test'})])
        self.assertEqual(self.app.events(), before)

    def test_bad_origin_never_reaches_upstream(self):
        self.connect_upstream()
        self.assertEqual(self.request('/api/demo/webcam/linq', {}, 'https://evil.invalid')[0], 403)
        self.assertEqual(self.seen, [])

    def test_payload_bound_before_forwarding(self):
        self.connect_upstream()
        self.assertEqual(self.request('/api/demo/webcam/linq', {'image':'a'*(4*1024*1024)})[0], 413)
        self.assertEqual(self.seen, [])


if __name__ == '__main__': unittest.main()
