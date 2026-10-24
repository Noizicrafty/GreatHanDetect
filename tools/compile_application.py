"""Compile one application model using the vendor recipe verified on Tiny YOLOv2."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
import os
os.environ.setdefault('MPLBACKEND','Agg')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from greathan.config import load_config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    parser.add_argument("--model",default="person")
    parser.add_argument("--calibration",required=True)
    parser.add_argument("--out",required=True)
    args=parser.parse_args()
    if platform.system()!='Linux' or sys.version_info[:3]!=(3,12,11):
        raise SystemExit('Use isolated Linux x86_64 Python 3.12.11 with vendor SDK 2026.9.45')
    import numpy as np
    import onnx
    from tpu_framework import RegularModel,Network,TPU_H1M,O1,TpuProgram,compiler,preliminary_model_conditioning
    spec=load_config(args.config)['models'][args.model]
    output=Path(args.out).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    source=Path(spec['onnx'])
    graph=onnx.load(str(source));onnx.checker.check_model(graph)
    print('1/5 Conditioning ONNX',flush=True)
    optimized=preliminary_model_conditioning(graph)
    onnx.save(optimized,str(output.with_suffix('.optimized.onnx')))
    names=[x.name for x in optimized.graph.input]
    calibration=np.load(args.calibration)
    tensors={name:calibration[name].astype(np.float32) for name in names}
    print('Calibration:',{k:v.shape for k,v in tensors.items()},flush=True)
    print('2/5 Calibrating with torch backend',flush=True)
    regular=RegularModel(model=optimized)
    thresholds=regular.calibrate_torch(calibration_data=tensors)
    RegularModel.dump_thresholds(str(output.with_suffix('.thresholds.json')),thresholds)
    print('3/5 Quantizing',flush=True)
    quantized=regular.quantize(thresholds)
    quantized.save(file_dir=str(output.parent),file_name=output.stem+'.qm.onnx')
    print('4/5 Building network',flush=True)
    network,renaming=Network.from_quantized_model(quantized)
    network.set_batch(1)
    print('5/5 Compiling TPU_H1M / O1',flush=True)
    executable,descriptions=compiler.compile_(hardware_parameters=TPU_H1M,network=network,parameters=O1)
    TpuProgram.from_executable(executable,descriptions).to_file(str(output))
    manifest={'status':'compiled_runtime_not_verified','program_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
              'onnx_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'target':'TPU_H1M','optimization':'O1',
              'inputs':spec['inputs'],'outputs':spec['outputs'],'renaming':renaming,
              'tensor_description_names':list(descriptions),'calibration_sha256':hashlib.sha256(Path(args.calibration).read_bytes()).hexdigest(),
              'model':args.model,'source_onnx':str(source),'program':str(output)}
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(output,manifest['program_sha256'],flush=True)


if __name__=='__main__':main()
