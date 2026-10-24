# GreatHanDetect

Video safety monitoring prototype for **LinQ TPU Framework 2026.9.45 / TPU_H1M**: inference, tracking, temporal rules, evidence, a persistent event journal and a Russian operator console.

**[Русская документация и запуск →](README_RU.md)** · [Documentation index](docs/README.md) · [Verified status](docs/BUILD_STATUS.md) · [Application guide](docs/APPLICATION_RU.md)

## Current state — 24 October 2026

- **Verified end to end:** people in a configured danger zone using actual remote LinQ AMQP inference, evidence, deduplicated episodes and browser notifications.
- **Implemented but not accepted:** smoking pipeline using object detection, pose and temporal heuristics. YOLO-World and YOLO11n-Pose compile and execute on LinQ; the quantized YOLO-World detector currently loses detections against its CPU reference.
- **Separate trained detector:** YOLOv8n trained for 26 completed epochs before the run was stopped. The illustrations below use that checkpoint through local PyTorch inference. These weights have not been integrated into the application or validated on TPU.
- **DEMO only:** helmet, vest and fall. No corresponding model executes; simulated events are marked and excluded from real-event metrics.
- **59 automated checks passed locally.** A 10-minute real-LinQ transport/application run was completed on repeated public still images. Real CCTV accuracy and production readiness remain unverified.

## Console demo without models or VPN

Install Python 3.12 and Git, clone the repository and run from its root:

```bash
python -m venv .venv
# Activate .venv for your shell.
python -m pip install -r requirements/application.txt
python tools/make_app_demo.py
python -m greathan --config assets/demo/replay_config.json --data results/qa
```

Open **http://127.0.0.1:8080/** and start processing. This explicitly marked replay checks the application, not a neural network. Windows: `setup_application_windows.bat`, `START_APPLICATION_WINDOWS.bat`. Linux: `bash setup_application.sh`, `bash start_application.sh`.

For actual inference, obtain the separate artifacts and follow [deployment instructions](docs/DEPLOYMENT_RU.md). Git contains source, tests, documentation, compact evidence and real prediction illustrations. Weights, ONNX/TPU binaries, datasets, proprietary SDK packages, VPN profiles and credentials are supplied separately.

## Actual predictions after 26 epochs

Nine test images selected with seed 42. Boxes and confidence values are direct model outputs at threshold 0.25. Examples include missed objects; these are not event-accuracy measurements.

![Actual predictions after 26 epochs](docs/images/predictions_epoch26.jpg)

[Checkpoint and inference provenance](docs/TRAINED_MODEL_RESULTS_RU.md) · [Measured LinQ results](docs/INTERMEDIATE_REPORT_RU.md) · [Remaining acceptance work](docs/ACCEPTANCE_CHECKLIST.md)

## Architecture

```text
Video -> LinQ AMQP / ONNX CPU -> decode + NMS -> IoU tracks
      -> danger-zone or smoking temporal rule
      -> episode + evidence -> SQLite + notifications -> operator browser
```

`greathan/` is the application; `tools/` contains preparation, calibration, compilation, reference inference and evaluation utilities. Older direct-device scripts remain for engineering work and are separate from the remote-AMQP application. [Architecture and API](docs/ARCHITECTURE_RU.md).

## Tests

```bash
python -m pip install -r requirements/runtime.txt
python tools/bootstrap_source.py
python -m compileall -q greathan tools tests
python -m unittest discover -s tests -v
```

The upstream smoking rule is fetched separately at commit `e7bba89b0336119363cf4d9f9e31b2e4a67e9233`. Tests do not require model binaries or TPU access. Training is optional and currently stopped; setup and CI never start training.

## Attribution and limits

The project adapts [clyv/smoking-detection](https://github.com/clyv/smoking-detection). Images/data: [Richie Lab / smoking-tasfx v4](https://universe.roboflow.com/richie-lab/smoking-tasfx/dataset/4), CC BY 4.0 according to the supplied export. [Third-party notices](docs/THIRD_PARTY_NOTICES.md). No blanket relicensing of upstream code, Ultralytics, models, datasets or SDK is claimed.

No verified 30 FPS, multi-camera production deployment or accepted smoking accuracy on industrial CCTV are claimed.
