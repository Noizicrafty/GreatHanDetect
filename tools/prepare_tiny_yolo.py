"""Normalize Model Zoo Tiny YOLOv2 ONNX metadata and generate a strict application config."""
import argparse
import hashlib
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import onnx
from greathan.config import ROOT, load_config, public_config, save_config


def prepare(source,output):
    model = onnx.load(str(source))
    initialized = {i.name for i in model.graph.initializer}
    inputs = [i for i in model.graph.input if i.name not in initialized]
    del model.graph.input[:]
    model.graph.input.extend(inputs)
    if len(inputs)!=1 or len(model.graph.output)!=1:
        raise ValueError("Unexpected Tiny YOLOv2 model")
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_value=1
    model.graph.output[0].type.tensor_type.shape.dim[0].dim_value=1
    model.ir_version=max(model.ir_version,4)
    onnx.checker.check_model(model)
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    onnx.save(model,str(output))
    config = public_config(load_config(ROOT/"configs/application.yaml"))
    config["models"]["person"]={
        "onnx":str(output.resolve()),"local_program":"assets/tpu/tiny_person.tpu",
        "remote_program":"/auto/cache/TEAM/tiny_person.tpu","input_size":416,"input_scale":255,
        "decoder":"tiny_yolov2","confidence":.3,"iou":.5,
        "inputs":{model.graph.input[0].name:{"shape":[1,3,416,416],"dtype":"float32"}},
        "outputs":{model.graph.output[0].name:{"shape":[1,125,13,13],"dtype":"float32"}},
        "source_url":"https://github.com/onnx/models/tree/main/validated/vision/object_detection_segmentation/tiny-yolov2",
        "onnx_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    path = output.with_suffix(".application.json")
    save_config(path,config)
    return path


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",required=True)
    parser.add_argument("--out",default="assets/onnx/tiny_person.onnx")
    args=parser.parse_args()
    print(prepare(args.source,args.out))
