from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time

from .backends import AmqpLinQBackend, OnnxBackend
from .contracts import Observation, Person
from tools.common import letterbox
from tools.decoder import decode_objects, decode_pose


class Vision:
    def __init__(self, config):
        self.config = config
        self.kind = config["backend"]
        self.models = config.get("models",{})
        self.frames = None
        if self.kind == "replay":
            self.frames = json.loads(Path(config["replay_file"]).read_text(encoding="utf-8"))
            self.backend = None
            self.version = "REPLAY_TEST_DATA_NO_NEURAL_INFERENCE"
        else:
            try:
                self.backend = AmqpLinQBackend(config["amqp"]) if self.kind == "linq_amqp" else OnnxBackend()
            except ImportError:
                raise ValueError("Cloud client dependencies missing: install pika and organiser tpu_pycloud_format wheel") from None
            selected = ["person"] if config["scenario"] == "danger_zone" else ["detector","pose"]
            versions = []
            for name in selected:
                spec = self.models[name]
                file = Path(spec.get("onnx", "")) if self.kind == "onnx_cpu" else Path(spec.get("local_program", ""))
                if file.is_file():
                    digest = hashlib.sha256()
                    with file.open("rb") as stream:
                        for chunk in iter(lambda:stream.read(1024*1024),b""):
                            digest.update(chunk)
                    sha = digest.hexdigest()
                    expected = spec.get("onnx_sha256") if self.kind == "onnx_cpu" else spec.get("program_sha256")
                    if expected and expected != sha:
                        raise ValueError(f"Model hash differs for {name}")
                    versions.append(name+":"+sha)
                else:
                    if self.kind == "onnx_cpu":
                        raise ValueError(f"Missing ONNX model: {name}")
                    versions.append(name+":remote-hash-unverified:"+spec.get("program_sha256","unspecified"))
            self.version = ";".join(versions)

    def _model(self, name, frame):
        spec = self.models[name]
        arr, meta = letterbox(frame, int(spec["input_size"]))
        if spec.get("input_scale",1) != 1:
            arr *= float(spec["input_scale"])
        outputs,timing = self.backend.infer(spec,{next(iter(spec["inputs"])):arr})
        raw = outputs[next(iter(spec["outputs"]))]
        confidence = float(spec.get("confidence",.3))
        iou = float(spec.get("iou",.5))
        if spec["decoder"] == "yolo_pose":
            persons = [Person(list(map(float,box)),float(score),points,kconf)
                       for box,score,points,kconf in decode_pose(raw,meta,confidence,iou)]
            return persons,[],timing
        if spec["decoder"] in ("yolo_objects","tiny_yolov2"):
            if spec["decoder"] == "tiny_yolov2":
                from .tiny_yolo import decode_tiny_yolo
                objects = decode_tiny_yolo(raw,meta,int(spec["input_size"]),confidence,iou)
            else:
                objects = decode_objects(raw,meta,spec["class_names"],confidence,iou)
            persons = [Person(list(map(float,box)),score) for label,score,box in objects if label == "person"]
            return persons,objects,timing
        raise ValueError("Unsupported decoder; supply a verified CPU-tail adapter for split graphs")

    def observe(self, frame, frame_id):
        if self.kind == "replay":
            values = self.frames.get(str(frame_id),{})
            return Observation([Person(**p) for p in values.get("persons",[])],values.get("objects",[]),{"replay":True})
        started = time.perf_counter()
        if self.config["scenario"] == "danger_zone":
            persons,objects,timing = self._model("person",frame)
            return Observation(persons,objects,{"person":timing,"total_s":time.perf_counter()-started})
        _,objects,detector_timing = self._model("detector",frame)
        persons,_,pose_timing = self._model("pose",frame)
        return Observation(persons,objects,{"detector":detector_timing,"pose":pose_timing,"total_s":time.perf_counter()-started})

    def close(self):
        if self.backend:
            self.backend.close()
