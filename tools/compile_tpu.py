"""Vendor TPU Framework 2026.9.45 ONNX -> PTQ -> TPU_H1M O1 -> .tpu.

Requires vendor SDK packages installed separately on Ubuntu; NOT bundled here.
Does not patch or silently rewrite unsupported ONNX operators.
"""
from __future__ import annotations
import argparse
import json
import sys
import traceback
from pathlib import Path
import numpy as np
from common import dest,project_cfg


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--part',choices=['detector','pose'],required=True)
    ap.add_argument('--skip-onnxruntime-check',action='store_true')
    args=ap.parse_args()
    if sys.platform!='linux':
        raise SystemExit('TPU SDK documentation requires Ubuntu 18.04+ (Linux). Run ONNX export on Windows or Ubuntu.')
    try:
        import onnx
        from tpu_framework import RegularModel,Network,TPU_H1M,O1,TpuProgram,compiler
    except ImportError as e:
        raise SystemExit('Vendor SDK is not installed in this Python environment. Install the TPU Framework packages first. '+str(e))
    config=project_cfg()['models'][args.part]
    onnx_file=dest(config['onnx']); calib_file=dest(config['calibration']); outfile=dest(config['tpu'])
    if not onnx_file.exists(): raise SystemExit('Missing ONNX: '+str(onnx_file))
    if not calib_file.exists(): raise SystemExit('Missing calibration data: '+str(calib_file))
    graph=onnx.load(str(onnx_file)); onnx.checker.check_model(graph)
    data=np.load(calib_file)
    input_names=[x.name for x in graph.graph.input]
    inputs={k:data[k].astype(np.float32) for k in input_names}
    print('Input tensors:',{k:v.shape for k,v in inputs.items()})
    if any(arr.ndim!=4 or arr.shape[1]!=3 for arr in inputs.values()):
        raise SystemExit('Only float32 NCHW RGB images are supported by this integration.')
    if not args.skip_onnxruntime_check:
        import onnxruntime as ort
        session=ort.InferenceSession(str(onnx_file),providers=['CPUExecutionProvider'])
        ref=session.run(None, {k:v[:1] for k,v in inputs.items()})
        print('ONNX Runtime outputs:',[x.shape for x in ref])
    print('1/4 Parsing ONNX into TPU Framework RegularModel ...')
    regular=RegularModel(model=graph)
    anchors=regular.anchors_mapping
    mapped={anchors.get(k,k):v for k,v in inputs.items()}
    print('2/4 Calibrating TPU network ...')
    thresholds=regular.calibrate(mapped)
    outfile.parent.mkdir(parents=True,exist_ok=True)
    RegularModel.dump_thresholds(str(outfile.with_suffix('.thresholds.json')),thresholds)
    print('3/4 Quantizing ...')
    quant=regular.quantize(thresholds)
    quant.save(file_dir=str(outfile.parent),file_name=outfile.stem+'.qm')
    print('4/4 Compiling for TPU_H1M with O1 ...')
    network,renaming=Network.from_quantized_model(quant)
    network.set_batch(1)
    executable,descriptions=compiler.compile_(TPU_H1M,network,O1)
    TpuProgram.from_executable(executable,descriptions).to_file(str(outfile))
    print('TPU PROGRAM:',outfile)
    print('IMPORTANT: successful compile is not equivalent to accuracy or correct runtime postprocessing.')
    manifest={'part':args.part,'onnx':str(onnx_file),'tpu':str(outfile),
              'calibration':str(calib_file),'target':'TPU_H1M','optimizer':'O1','batch':1,
              'onnx_inputs':input_names,'onnx_outputs':[v.name for v in graph.graph.output],
              'status':'compiler_completed; runtime_not_verified'}
    outfile.with_suffix('.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

if __name__=='__main__':
    try:main()
    except Exception as e:
        print('\nTPU COMPILATION FAILED:',e,file=sys.stderr)
        print('See docs/COMPATIBILITY.md: YOLO-World / YOLO11-Pose are not on the certified list.',file=sys.stderr)
        traceback.print_exc()
        sys.exit(2)
