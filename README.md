# GreatHanDetect — smoking detection adapted for LinQ TPU

An **experimental engineering adapter** for [clyv/smoking-detection](https://github.com/clyv/smoking-detection) targeting **LinQ TPU Framework 2026.9.45 / TPU_H1M**.

[Русская инструкция](README_RU.md) · [Status](docs/BUILD_STATUS.md) · [Compatibility](docs/COMPATIBILITY.md) · [Acceptance criteria](docs/ACCEPTANCE_CHECKLIST.md)

> **Important:** This repository contains source code for porting and testing the model, **not** a certified, successfully compiled LinQ TPU application. **No original trained weights, ONNX binaries, calibration images, dataset files, TPU executables or proprietary vendor SDK are committed.** ONNX export, quantization and TPU runtime require testing with real artifacts and hardware.

## Pipeline

```text
Camera / MP4 / CCTV
 ├─ YOLO-World (cigarette, vape, cigar, pipe, smoke)
 │       -> ONNX -> PTQ -> LinQ TPU_H1M ──┐
 └─ YOLO11n-Pose (people/keypoints)        │
         -> ONNX -> PTQ -> LinQ TPU_H1M ───┤
                                          v
                   CPU: decode raw heads, NMS, IoU tracking
                   CPU: clyv smoking-action feature heuristics
                                          v
                                annotated.mp4 + events.json
```

YOLO-World and YOLO11n-Pose **are not listed as verified models** in the supplied LinQ SDK 2026.9.45 documentation; another architecture, `yolov8n_pose`, is listed with special truncated outputs. Compiler support and decoder compatibility are unverified.

## Windows: test original CPU/GPU model

Prerequisites: Git, Python 3.12 for the provided bootstrap scripts (the original upstream may support other versions).

```powershell
git clone https://github.com/Noizicrafty/GreatHanDetect.git
cd GreatHanDetect
.\setup_windows.bat
.\START_CPU_BASELINE_WINDOWS.bat
# Or use START_GPU_BASELINE_WINDOWS.bat on a CUDA-compatible PC
```

The baseline downloads the original clyv source at pinned upstream commit `e7bba89b0336119363cf4d9f9e31b2e4a67e9233`. Ultralytics retrieves standard pretrained weights on first use. **It is not a validated industrial smoking-action checkpoint.**

## Export static ONNX

```powershell
.\EXPORT_AND_INSPECT_WINDOWS.bat
```

Or in the activated Python environment:

```bash
python tools/bootstrap_source.py
python tools/export_onnx.py --part both --device cpu
python tools/inspect_onnx.py --part detector
python tools/inspect_onnx.py --part pose
python tools/run_onnx_reference.py --video test.mp4 --max-frames 100
```

The models use static batch=1, image size 640, FP32 NCHW RGB/255 with letterbox and no exported NMS.

## Ubuntu: LinQ TPU pipeline

LinQ documentation requires **Ubuntu 18.04+** and **Python 3.12.11**. Obtain and install the **vendor's SDK packets** separately.

```bash
bash setup_ubuntu.sh
bash INSTALL_VENDOR_PACKETS_UBUNTU.sh /path/to/vendor/packets
source .venv/bin/activate

python tools/bootstrap_source.py
python tools/export_onnx.py --part both --device cpu
python tools/inspect_onnx.py --part detector
python tools/inspect_onnx.py --part pose
python tools/make_calibration.py --part detector --images /data/cctv_frames --limit 64
python tools/make_calibration.py --part pose --images /data/cctv_frames --limit 64
bash COMPILE_BOTH_UBUNTU.sh
```

After successful **real compilation** and after a connected LinQ TPU is discovered via `pytpu.Device.list_devices()`:

```bash
python tools/run_tpu_video.py --video /data/test.mp4 --show --max-frames 100
```

Outputs: `results/tpu_pipeline/*_annotated.mp4` and `*_events.json`. The adapter employs an IoU tracker, **not** the original ByteTrack, and does not load an optional temporal classifier. Actual results depend on two TPU heads being supported in standard raw output form.

## Local checks

```bash
python -m pip install -r requirements/runtime.txt
python -m unittest discover -s tests -v
python tools/doctor.py
```

5 synthetic preprocessing/decoding unit tests were reported passing in the local preparation environment. This does **not** validate detection performance.

## Licensing

The upstream `clyv/smoking-detection` code is fetched separately, rather than copied, because its license and redistribution rights require verification. Ultralytics, weights, datasets and the vendor TPU SDK have separate licensing requirements.

## Performance and limitations

No demonstrated 30 FPS, no validated detection on low-quality CCTV, no confirmed successful compile/run on actual LinQ TPU. Measure smoking-event recall, false positives per hour, precision, latency, and sustained real FPS on your enterprise recordings before production use.
