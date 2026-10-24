"""Strict named-tensor transports. Cloud credentials are read from environment only."""
from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import time
from urllib.parse import quote
import uuid

from .contracts import validate_tensors


class BackendError(RuntimeError):
    pass


class AmqpLinQBackend:
    kind = "linq_amqp"

    def __init__(self, config, *, pika_module=None, codec=None):
        if pika_module is None:
            import pika as pika_module
        if codec is None:
            import tpu_pycloud_format as codec
        self.pika, self.codec = pika_module, codec
        self.config = config
        self.timeout = float(config.get("timeout_s", 30))
        if not 0 < self.timeout <= 120:
            raise ValueError("AMQP timeout must be in (0, 120]")
        self.connection = self.channel = None
        self.queue = None
        self.consumer = None

    def _connect(self):
        if self.connection is not None and self.connection.is_open:
            return
        user = os.environ.get(self.config.get("username_env", "LINQ_AMQP_USER"))
        password = os.environ.get(self.config.get("password_env", "LINQ_AMQP_PASSWORD"))
        if not user or not password:
            raise BackendError("Set LINQ_AMQP_USER and LINQ_AMQP_PASSWORD for the organiser broker")
        p = self.pika
        self.connection = p.BlockingConnection(p.ConnectionParameters(
            host=self.config["host"], port=int(self.config.get("port", 5672)),
            virtual_host=self.config.get("virtual_host", "/"),
            credentials=p.PlainCredentials(user, password),
            socket_timeout=min(10, self.timeout), stack_timeout=self.timeout,
            blocked_connection_timeout=self.timeout, heartbeat=30,
            connection_attempts=1))
        try:
            self.channel = self.connection.channel()
            declared = self.channel.queue_declare(queue="", exclusive=True, auto_delete=True)
            self.queue = declared.method.queue
            self.channel.queue_bind(exchange=self.config.get("exchange", "tpu.direct"),
                                    queue=self.queue, routing_key=self.queue)
            self.consumer = self.channel.consume(self.queue,auto_ack=True,inactivity_timeout=.1)
        except Exception:
            self.close()
            raise

    @staticmethod
    def program_uri(path):
        p = PurePosixPath(path)
        if not p.is_absolute() or ".." in p.parts or p.parts[:3] != ("/", "auto", "cache"):
            raise ValueError("Remote TPU program must be an absolute /auto/cache/... path")
        if p.suffix != ".tpu":
            raise ValueError("Remote program must end in .tpu")
        return "file://" + quote(str(p), safe="/") + "?raw=False"

    def infer(self, model, tensors):
        validate_tensors(tensors, model["inputs"], "input")
        uri = self.program_uri(model["remote_program"])
        total_started = time.perf_counter()
        codec = self.codec
        try:
            self._connect()
            request_id = uuid.uuid4().hex
            before_encode = time.perf_counter()
            body = codec.serialize(codec.PyTask(tensors))
            submitted = time.perf_counter()
            self.channel.basic_publish(
                exchange=self.config.get("exchange", "tpu.direct"),
                routing_key=self.config.get("routing_key", "input"), body=body,
                properties=self.pika.BasicProperties(headers={
                    codec.TARGET_URI_ATTR_NAME: uri,
                    codec.RESPONSE_ROUTING_KEYS_ATTR_NAME: json.dumps([self.queue]),
                    codec.MESSAGE_ID_ATTR_NAME: request_id}))
            deadline = total_started + self.timeout
            ignored = 0
            while time.perf_counter() < deadline:
                method, properties, response = next(self.consumer)
                if method is None:
                    continue
                if (properties.headers or {}).get(codec.MESSAGE_ID_ATTR_NAME) != request_id:
                    ignored += 1
                    continue
                received = time.perf_counter()
                outputs, metadata, worker = codec.deserialize(response)
                if worker.get("status_code") != 0:
                    # Worker messages can include paths or credentials; keep diagnostics bounded and generic.
                    raise BackendError(f"LinQ worker failed with status {worker.get('status_code')}")
                validate_tensors(outputs, model["outputs"], "output")
                finished = time.perf_counter()
                return outputs, {
                    "request_id": request_id, "backend": self.kind,
                    "connect_s": before_encode-total_started,
                    "serialize_s": submitted-before_encode,
                    "request_reply_s": received-submitted,
                    "deserialize_validate_s": finished-received,
                    "total_s": finished-total_started,
                    "request_bytes": len(body), "response_bytes": len(response),
                    "ignored_responses": ignored,
                    "worker_pid": str(worker.get("worker_pid", "")),
                    "worker_inference_time_raw": worker.get("inference_time"),
                }
            raise BackendError(f"LinQ response timeout after {self.timeout:g} s")
        except (ValueError, BackendError):
            self.close()
            raise
        except Exception as exc:
            self.close()
            raise BackendError(f"AMQP transport failed ({type(exc).__name__}); verify VPN/broker") from None

    def close(self):
        connection, self.connection = self.connection, None
        self.channel = self.queue = self.consumer = None
        if connection is not None and connection.is_open:
            try:
                connection.close()
            except Exception:
                pass


class OnnxBackend:
    kind = "onnx_cpu"

    def __init__(self, config=None):
        self.sessions = {}

    def infer(self, model, tensors):
        import onnxruntime as ort
        validate_tensors(tensors, model["inputs"], "input")
        path = str(Path(model["onnx"]).resolve())
        if path not in self.sessions:
            self.sessions[path] = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        started = time.perf_counter()
        session = self.sessions[path]
        names = [x.name for x in session.get_outputs()]
        outputs = dict(zip(names, session.run(None, tensors)))
        validate_tensors(outputs, model["outputs"], "output")
        return outputs, {"backend": self.kind, "total_s": time.perf_counter()-started}

    def close(self):
        self.sessions.clear()
