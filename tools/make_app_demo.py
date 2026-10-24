"""Generate video + replay detections for application QA, never model validation."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from greathan.config import ROOT, load_config, save_config, public_config


def generate(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    fps, width, height, duration = 10, 960, 540, 20
    video = directory/"demo.mp4"
    writer = cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,height))
    if not writer.isOpened():
        raise RuntimeError("Cannot create QA video")
    frames = {}
    try:
        for index in range(fps*duration):
            t = index/fps
            frame = np.full((height,width,3),(35,43,52),dtype=np.uint8)
            for x in range(0,width,60):
                cv2.line(frame,(x,0),(x,height),(45,53,62),1)
            for y in range(0,height,60):
                cv2.line(frame,(0,y),(width,y),(45,53,62),1)
            inside = 3 <= t < 10 or 14 <= t < 19
            visible = not 6 <= t < 6.6
            x = 570 if inside else 170
            people = []
            if visible:
                box = [x,140,x+100,440]
                people.append({"box":box,"confidence":.95})
                cv2.rectangle(frame,(x,140),(x+100,440),(185,175,115),-1)
                cv2.circle(frame,(x+50,110),30,(185,175,115),-1)
            cv2.putText(frame,"QA REPLAY - SYNTHETIC VIDEO, NO MODEL VALIDATION",(28,35),cv2.FONT_HERSHEY_SIMPLEX,.65,(220,220,220),1)
            cv2.putText(frame,f"Source time: {t:.1f}s",(28,510),cv2.FONT_HERSHEY_SIMPLEX,.6,(220,220,220),1)
            frames[str(index+1)] = {"persons":people}
            writer.write(frame)
    finally:
        writer.release()
    replay = directory/"detections.json"
    replay.write_text(json.dumps(frames),encoding="utf-8")
    config = public_config(load_config(ROOT/"configs/application.yaml"))
    config.update(backend="replay",replay_file=str(replay))
    config['analysis_fps']=5  # Fixed QA cadence, independent from production defaults.
    config["camera"].update(source=str(video),mode="realtime",id="qa-camera")
    output = directory/"replay_config.json"
    save_config(output,config)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",default="assets/demo")
    args = parser.parse_args()
    print(generate(args.out))
