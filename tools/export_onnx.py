"""Export the same clyv detector & pose networks as frozen, static ONNX for separate TPU compilation."""
import argparse
import json
import shutil
from pathlib import Path
from common import ROOT, project_cfg, dest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--part', choices=['both', 'detector', 'pose'], default='both')
    p.add_argument('--opset', type=int, default=12)
    p.add_argument('--device', default='cpu', help='cpu or GPU index; exporter only')
    args = p.parse_args()
    from ultralytics import YOLO, YOLOWorld
    cfg = project_cfg()['models']
    for part in (['detector', 'pose'] if args.part == 'both' else [args.part]):
        section = cfg[part]
        w = section['weights']
        if part == 'detector' and section.get('type', 'yoloworld') == 'yoloworld':
            model = YOLOWorld(w)
            model.set_classes(section['class_names'])
        else:
            model = YOLO(w)
            if part == 'detector':
                expected = [str(x) for x in section['class_names']]
                actual = [str(model.names[i]) for i in sorted(model.names)]
                if actual != expected:
                    raise ValueError(f'Detector class order mismatch. Expected {expected}, got {actual}')
        output = Path(model.export(format='onnx', imgsz=section['input_size'], opset=args.opset,
                                   dynamic=False, batch=1, nms=False, simplify=False,
                                   half=False, device=args.device))
        target = dest(section['onnx'])
        target.parent.mkdir(parents=True, exist_ok=True)
        if output.resolve() != target.resolve():
            shutil.copy2(output, target)
        meta = {'part':part, 'weights':w, 'input_size':section['input_size'],
                'classes':section.get('class_names'), 'onnx':str(target),
                'postprocessing':'external', 'nms_exported':False}
        target.with_suffix('.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        print(f'EXPORTED: {target} ({target.stat().st_size/1024**2:.1f} MiB)')
        print('IMPORTANT: exporter success does not guarantee TPU compiler compatibility.')


if __name__ == '__main__':
    main()
