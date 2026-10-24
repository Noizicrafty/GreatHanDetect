"""Check ONNX graph and CPU inference before TPU conversion."""
import argparse
import json
import numpy as np
from common import project_cfg, dest

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', choices=['detector','pose'], required=True)
    args=ap.parse_args()
    import onnx
    import onnxruntime as ort
    spec=project_cfg()['models'][args.part]
    file=dest(spec['onnx'])
    model=onnx.load(str(file))
    onnx.checker.check_model(model)
    sess=ort.InferenceSession(str(file),providers=['CPUExecutionProvider'])
    assert len(sess.get_inputs()) == 1, 'Expected single ONNX input.'
    inp=sess.get_inputs()[0]
    assert inp.type == 'tensor(float)', f'Expected FP32, got {inp.type}'
    expected=[1,3,spec['input_size'],spec['input_size']]
    if inp.shape != expected: raise SystemExit(f'Unexpected ONNX shape {inp.shape}, expected {expected}')
    outputs=sess.run(None,{inp.name:np.full(expected,0.5,np.float32)})
    print('Outputs:',json.dumps({x.name:list(a.shape) for x,a in zip(sess.get_outputs(),outputs)},indent=2))
    if not any(a.ndim==3 for a in outputs):
        print('WARNING: raw YOLO heads absent; decoder may require adaptation.')

if __name__=='__main__':main()
