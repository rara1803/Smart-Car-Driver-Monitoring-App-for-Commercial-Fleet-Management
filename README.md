# Lane Deviation Pipeline — Smart Car & Driver Monitoring App

External (road-facing) camera pipeline for a commercial fleet driver-safety system. This repo covers the piece I'm responsible for within our capstone: real-time lane detection and lateral deviation scoring, running on an NVIDIA Jetson Orin Nano and feeding into a broader "Driver Reliability Score" (Rscore) computed alongside an internal drowsiness/emotion detection pipeline.

## Project Context

The Smart Car and Driver Monitoring App is a closed-loop driver safety system built on a 1/10 scale RC car prototype. Two AI pipelines run in parallel on a Jetson Orin Nano:

- **Internal pipeline** (teammate's responsibility): ResNet-18 based drowsiness and emotion detection via a driver-facing camera.
- **External pipeline** (this repo): lane detection and lateral deviation via YOLOPv2, fine-tuned on a custom tape-on-cardboard track and exported to TensorRT for real-time inference.

Both pipelines feed a live Rscore (starting at 100) to an Android app over WebSocket/REST, built by the software team, which displays the score to the driver and logs sessions for fleet admins.

## What's in this repo

- **`demo_trt.py`** — the deployed, real-time inference script. Loads the TensorRT-optimized YOLOPv2 engine, runs it against a live IMX219 CSI camera feed, computes a smoothed [0, 1] lateral deviation score, overlays the lane geometry on screen, and writes the live score to a file-based interface for the fleet server integration. Includes an HSV-based backup lane detector (for when the primary model loses the tape lines) and a built-in calibration mode.
- **`carla_manual_validation.py`** — a CARLA simulator harness used to validate the lane deviation logic (and a hybrid actor/sensor-based obstacle detector) before physical deployment, with manual keyboard driving and JSONL telemetry logging.

## Pipeline Summary

1. **Fine-tune** YOLOPv2's lane segmentation head on a custom RC-car track dataset (yellow tape on black cardboard).
2. **Export** the fine-tuned model to TensorRT FP16 for real-time inference on the Jetson.
3. **Infer** on live camera frames; extract lane line geometry from the segmentation mask (`extract_lane_lines`), with an HSV-threshold fallback for when the primary model's confidence drops.
4. **Compute deviation**: the vehicle's offset from lane center, normalized to [0, 1] by lane width, with a deadzone and exponential moving average smoothing.
5. **Expose** the score via a file-based interface for the fleet server / Rscore fusion logic.
6. **Validate** the same math in CARLA before trusting it on hardware.

## Hardware / Software Environment

| Component | Version |
|---|---|
| Device | NVIDIA Jetson Orin Nano |
| JetPack | 6.1 (L4T R36.4.7) |
| CUDA | 12.6 |
| cuDNN | 9.3.0 |
| Python | 3.10 |
| OS | Ubuntu 22.04 (aarch64) |
| torch | 2.5.0a0+872d972e41.nv24.08 (NVIDIA Jetson official wheel) |
| torchvision | 0.20.0a0+afc54f7 (built from source) |
| numpy | 1.26.4 (must stay below 2.0) |
| cuSPARSELt | 0.7.1 (installed via tegra deb) |

`carla_manual_validation.py` was developed and run on a desktop machine with CARLA installed, not on the Jetson itself.

## Status of this repo

**This repo contains the pipeline source code, not a runnable package.** The fine-tuned weights (`yolopv2.pt`), the exported TensorRT engine (`yolopv2_full.engine`), and the `utils/utils.py` helper module were stored only on the Jetson Orin Nano used for development. That device was later reset by another user before those files could be backed up, so they are not recoverable and are not included here.

What that means in practice:

- `demo_trt.py` and `carla_manual_validation.py` are complete, working reference implementations of the lane detection and deviation-scoring logic, export pipeline usage, and CARLA validation harness.
- To actually run either script, you would need to: (1) fine-tune YOLOPv2 on your own track dataset following the same procedure (segmentation head fine-tuning on tape-on-cardboard imagery), (2) export to TensorRT following NVIDIA's standard ONNX → `onnxsim` → TensorRT engine workflow, and (3) obtain `utils/utils.py` from the original [YOLOPv2](https://github.com/CAIC-AD/YOLOPv2) repo (it supplies `LoadCamera`, `lane_line_mask`, `driving_area_mask`, etc. — used largely unmodified).
- A recorded demo video of the working pipeline on hardware exists separately; see the Demo section below.

## Expected Project Structure

If reconstructed with the missing pieces above, the layout the scripts expect is:

```
.
├── demo_trt.py                    # this repo
├── carla_manual_validation.py     # this repo
├── utils/
│   └── utils.py                   # from YOLOPv2 (LoadCamera, lane_line_mask, etc.) — not included, see Status
├── data/
│   └── weights/
│       └── yolopv2_full.engine    # fine-tuned, TensorRT-exported engine — not included, see Status
├── models/
│   └── yolopv2.pt                 # TorchScript weights (used by the CARLA script) — not included, see Status
└── outputs/                       # CARLA logs + debug frames (created at runtime)
```

## Running

**On the Jetson (real hardware):**
```bash
python3 demo_trt.py --exist-ok
```

**Calibrating the reference lane center/width for a new track:**
```bash
python3 demo_trt.py --calibrate --calib-frames 60
```

**CARLA validation (on a desktop with CARLA running):**
```bash
python3 carla_manual_validation.py
```

## Demo

A recorded video of the pipeline running live on the RC car hardware is available [here](#) *(replace with your upload link — YouTube unlisted or Google Drive works well since the file is large)*.

## Notes

- `demo_trt.py` writes the live deviation score to `/tmp/lane_deviation.txt` as a simple file-based interface for other components (e.g. the fleet server bridge) to read.
- The lane detector runs YOLOPv2 as primary and an HSV yellow-tape threshold as backup — both produce a binary mask in the same coordinate space, so the deviation math is identical regardless of source.
- `carla_manual_validation.py` streams telemetry to the Jetson over WebSocket only if `ENABLE_JETSON_WEBSOCKET = True`; update `JETSON_WS_URL` with your device's actual address before enabling it.
