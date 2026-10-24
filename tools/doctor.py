"""Non-destructive environment and artifact check."""
from pathlib import Path
import importlib.util
import platform
import sys
from common import ROOT,project_cfg,dest

def status(label,val):
    print(('[ OK ]' if val else '[----]')+' '+label)

cfg=project_cfg()
print('Python',sys.version.split()[0],'OS',platform.platform())
status('Python 3.12.11 for TPU SDK',sys.version_info[:3]==(3,12,11))
status('Linux/Ubuntu for TPU SDK',sys.platform=='linux')
for module in ('torch','ultralytics','onnx','onnxruntime','cv2','pytpu','tpu_framework'):
    try: ok=importlib.util.find_spec(module) is not None
    except Exception: ok=False
    status(module,ok)
status('clyv source',(ROOT/'vendor/smoking-detection/smokedet/fusion.py').exists())
for part,spec in cfg['models'].items():
    for key in ('onnx','calibration','tpu'):
        path=dest(spec[key])
        status(f'{part} {key}: {path}',path.is_file())
status('TPU device node',bool(list(Path('/dev').glob('tpu*'))) if sys.platform=='linux' else False)
print('Checks are not TPU model certification.')
