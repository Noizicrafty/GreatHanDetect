"""Build calibration tensors for one application model using its exact preprocessing."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from greathan.config import load_config
from tools.common import letterbox


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    parser.add_argument("--model",default="person")
    parser.add_argument("--images",nargs="+",required=True)
    parser.add_argument("--out",required=True)
    parser.add_argument("--augment",type=int,default=1,help="Variants per source image; these are not independent validation samples")
    args=parser.parse_args()
    model=load_config(args.config)["models"][args.model]
    arrays=[];sources=[]
    for value in args.images:
        path=Path(value)
        frame=cv2.imread(str(path))
        if frame is None:raise ValueError("Cannot read calibration image: "+str(path))
        sources.append({"file":path.name,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
        for index in range(args.augment):
            variant=frame if index==0 else cv2.convertScaleAbs(frame,alpha=.7+.1*index,beta=0)
            arr,_=letterbox(variant,int(model["input_size"]))
            arr*=float(model.get("input_scale",1))
            arrays.append(arr[0])
    tensor=np.stack(arrays).astype(np.float32)
    output=Path(args.out);output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,**{next(iter(model["inputs"])):tensor})
    output.with_suffix(".json").write_text(json.dumps({"sources":sources,"variants_per_source":args.augment,"shape":list(tensor.shape),"purpose":"calibration_only_not_validation"},indent=2),encoding="utf-8")
    print(output,tensor.shape)


if __name__=="__main__":main()
