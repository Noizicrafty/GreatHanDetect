from __future__ import annotations
import copy
import json
from pathlib import Path

from .contracts import config_hash, finite_number

ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    path = Path(path).resolve()
    import yaml
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = validate_config(config)
    for model in config.get("models", {}).values():
        for key in ("onnx","local_program"):
            if model.get(key) and not Path(model[key]).is_absolute():
                model[key] = str((ROOT/model[key]).resolve())
    if config.get("replay_file") and not Path(config["replay_file"]).is_absolute():
        config["replay_file"]=str((ROOT/config["replay_file"]).resolve())
    camera=config["camera"]
    if "://" not in camera["source"] and not camera["source"].isdecimal() and not Path(camera["source"]).is_absolute():
        camera["source"]=str((ROOT/camera["source"]).resolve())
    config["_path"] = str(path)
    return config


def validate_config(config):
    if not isinstance(config,dict):
        raise ValueError("Configuration must be a mapping")
    config = copy.deepcopy(config)
    demo=config.get('demo_additional',{})
    if not isinstance(demo,dict) or not isinstance(demo.get('enabled',False),bool):
        raise ValueError('demo_additional must have a boolean enabled flag')
    from .capabilities import EXTRA_SCENARIOS
    if not isinstance(demo.get('scenarios',[]),list) or any(x not in EXTRA_SCENARIOS for x in demo.get('scenarios',[])):
        raise ValueError('Unknown demonstration scenario')
    if len(demo.get('scenarios',[]))!=len(set(demo.get('scenarios',[]))):
        raise ValueError('Demonstration scenarios must be unique')
    for key in ("password","credentials","token","api_key"):
        if key in config.get("amqp",{}):
            raise ValueError("Keep broker credentials in environment variables, not configuration files")
    if config.get("backend") not in ("onnx_cpu","linq_amqp","replay"):
        raise ValueError("backend must be onnx_cpu, linq_amqp or replay")
    if config.get("scenario") not in ("danger_zone","smoking"):
        raise ValueError("scenario must be danger_zone or smoking")
    camera = config.setdefault("camera",{})
    if not isinstance(camera.get("id"),str) or not camera["id"].strip():
        raise ValueError("camera.id is required")
    if not isinstance(camera.get("source"),str) or not camera["source"].strip():
        raise ValueError("camera.source is required")
    if camera.get("mode","offline") not in ("offline","realtime","live"):
        raise ValueError("camera.mode must be offline, realtime or live")
    finite_number(config.setdefault("analysis_fps",5),"analysis_fps",.1,60)
    rules = config.setdefault("rules",{})
    for key,default,minimum in (("confirm_s",1.5,0),("clear_s",2,0),("missing_s",3,.1)):
        rules[key] = finite_number(rules.get(key,default),key,minimum,120)
    rules["min_samples"] = int(finite_number(rules.get("min_samples",3),"min_samples",1,1000))
    zones = config.setdefault("zones",[])
    if config["scenario"] == "danger_zone" and not zones:
        raise ValueError("Configure at least one danger zone")
    ids = []
    for zone in zones:
        if not isinstance(zone.get("id"),str) or not zone["id"]:
            raise ValueError("Every zone requires an id")
        ids.append(zone["id"])
        points = zone.get("polygon",[])
        if not 3 <= len(points) <= 100:
            raise ValueError("Zone polygon requires 3..100 points")
        for point in points:
            if len(point) != 2:
                raise ValueError("Polygon points must be [x,y]")
            for v in point:
                finite_number(v,"polygon coordinate",0,1)
        area = abs(sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(points,points[1:]+points[:1])))/2
        if area < 1e-5:
            raise ValueError("Zone polygon has zero area")
    if len(ids) != len(set(ids)):
        raise ValueError("Zone IDs must be unique")
    # Never persist passwords or URLs embedding credentials in runtime snapshots.
    source = camera["source"]
    from urllib.parse import urlsplit
    parsed = urlsplit(source) if "://" in source else None
    if parsed and (parsed.username or parsed.password):
        raise ValueError("Use a source environment variable for authenticated stream URLs")
    if config["backend"] != "replay":
        required = ["person"] if config["scenario"] == "danger_zone" else ["detector","pose"]
        for name in required:
            model = config.get("models",{}).get(name)
            if not model:
                raise ValueError(f"Missing model specification: {name}")
            for key in ("inputs","outputs","decoder","input_size"):
                if not model.get(key):
                    raise ValueError(f"{name}.{key} is required")
            size=int(finite_number(model["input_size"],"input_size",32,4096))
            if model["decoder"] not in ("yolo_objects","yolo_pose","tiny_yolov2"):
                raise ValueError("Unsupported decoder")
            finite_number(model.get("confidence",.3),"confidence",0,1)
            finite_number(model.get("iou",.5),"iou",0,1)
            finite_number(model.get("input_scale",1),"input_scale",.001,255)
            if model["decoder"]=="yolo_objects" and not model.get("class_names"):
                raise ValueError("Object detector requires ordered class_names")
            for specs in (model["inputs"],model["outputs"]):
                for tensor_name,spec in specs.items():
                    if not isinstance(tensor_name,str) or not tensor_name or not all(isinstance(n,int) and n>0 for n in spec["shape"]):
                        raise ValueError("Tensor manifests require named static positive shapes")
            if len(model["inputs"]) != 1 or len(model["outputs"]) != 1:
                raise ValueError("Current YOLO decoders require one input and one assembled output; supply CPU-tail adapter for split graphs")
            if next(iter(model["inputs"].values()))["shape"] != [1,3,size,size]:
                raise ValueError("Image model input must be float32 [1,3,input_size,input_size]")
            if next(iter(model["inputs"].values())).get("dtype","float32") != "float32":
                raise ValueError("Image preprocessing produces float32")
            if config["backend"] == "linq_amqp":
                from .backends import AmqpLinQBackend
                AmqpLinQBackend.program_uri(model["remote_program"])
    return config


def public_config(config):
    return {k:v for k,v in config.items() if not k.startswith("_")}


def save_config(path, config):
    config = validate_config(public_config(config))
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf-8")
    temp.replace(path)
