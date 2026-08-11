#!/usr/bin/env python3
"""
demo_trt.py  (Jetson Orin Nano)
==================================
YOLOPv2 (TensorRT) = PRIMARY lane detector. HSV yellow-tape = BACKUP / reinforcement.

The deviation math is SOURCE-AGNOSTIC: it gets a binary mask and never knows whether
it came from YOLO or HSV. Both masks are produced in the SAME (model mask) coordinate
space, so the same math applies to both.

------------------------------------------------------------------------------
QUICK START  (type each command on ONE line)
------------------------------------------------------------------------------
  Normal run:
      python3 demo_trt.py --exist-ok

  Calibration run (finds --ref-center / --lane-width for your track):
      python3 demo_trt.py --calibrate --calib-frames 60
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import tensorrt as trt

from utils.utils import (
    time_synchronized,
    increment_path,
    driving_area_mask,
    lane_line_mask,
    show_seg_result,
    AverageMeter,
    LoadImages,
    LoadCamera,
)

# ══════════════════════════════════════════════════════════════════════════════
#  >>>>>  THE KNOBS YOU ACTUALLY TUNE  <<<<<
# ══════════════════════════════════════════════════════════════════════════════
# Calibration (MASK space). None -> sensible default (image center / auto width).
# Set these after you read a steady C= / W= from the printout, OR use --ref-center/--lane-width.
LANE_CENTER_CAL = None      # the "centered" reference x
LANE_WIDTH_CAL = None       # nominal lane width, used to estimate the missing edge

SMOOTH_ALPHA = 0.5          # EMA: score = ALPHA*new + (1-ALPHA)*prev
DEADZONE = 0.03             # deviations smaller than this snap to 0
DET_MODE = "both"           # "yolo" | "hsv" | "both"   (overridable with --mode)

# Plausible lane width as a fraction of mask width (replaces the old 100/240 HSV px range).
LANE_WIDTH_MIN_FRAC = 0.12
LANE_WIDTH_MAX_FRAC = 0.80
# ══════════════════════════════════════════════════════════════════════════════

USE_IMX_CAM = True
TRT_H, TRT_W = 384, 640

# Engine output tensor names (engine-specific; printed at startup). In demo_trt.py the
# tensor literally named "ll_seg" is the DRIVING AREA, and "759" is the LANE LINE.
DA_TENSOR_NAME = "ll_seg"   # driving area  (1,2,384,640)
LL_TENSOR_NAME = "759"      # lane line     (1,1,384,640)

# HSV backup (yellow tape)
HSV_LOWER = np.array([15, 60, 60])
HSV_UPPER = np.array([35, 255, 255])

# ROI as fraction of mask height (lower band = closest to the car)
ROI_TOP_FRAC = 0.35
ROI_BOTTOM_FRAC = 0.98

# Lane extraction
MIN_LANE_PIXELS = 120       # min lane pixels in ROI to attempt detection
MIN_BLOB_AREA = 12          # drop connected components smaller than this
MIN_SPLIT_GAP_FRAC = 0.10   # min x-gap (frac of width) to split left vs right line

CONF_RANK = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
REC_FPS = 6


# ══════════════════════════════════════════════════════════════════════════════
# TensorRT engine
# ══════════════════════════════════════════════════════════════════════════════
class TRTEngine:
    """Thin wrapper around a serialized TensorRT engine for lane/road segmentation."""

    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.tensors = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            self.tensors[name] = torch.empty(shape, dtype=torch.float16, device="cuda")
        print(f"[TRT] Engine loaded: {engine_path}")
        print(f"[TRT] IO tensors: {list(self.tensors.keys())}")

    def infer(self, img_tensor):
        self.tensors["images"].copy_(img_tensor)
        for name, t in self.tensors.items():
            self.context.set_tensor_address(name, t.data_ptr())
        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()
        return self.tensors


def _get_tensor(tensors, name):
    if name not in tensors:
        raise KeyError(
            f"Output tensor '{name}' not found. Available: {list(tensors.keys())}. "
            f"Update DA_TENSOR_NAME / LL_TENSOR_NAME at the top of this file."
        )
    return tensors[name]


# ══════════════════════════════════════════════════════════════════════════════
# Source-agnostic lane geometry (works on ANY binary mask: YOLO or HSV)
# ══════════════════════════════════════════════════════════════════════════════
def _line_x_at(line, y):
    """Evaluate a cv2.fitLine (vx, vy, x0, y0) at a given y, returning x."""
    vx, vy, x0, y0 = line
    if abs(vy) < 1e-6:
        return x0
    return x0 + (y - y0) * (vx / vy)


def _fit_line(labels, comp_list):
    """Fit a straight line through the pixels belonging to the given connected components."""
    if not comp_list:
        return None, 0
    label_ids = np.array([c[1] for c in comp_list], dtype=np.int32)
    sel = np.isin(labels, label_ids)
    ys, xs = np.where(sel)
    if len(xs) < 8:
        return None, len(xs)
    pts = np.column_stack((xs, ys)).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    return (float(vx), float(vy), float(x0), float(y0)), len(xs)


def _split_left_right(comps, w):
    """Split components into left/right groups at the largest x-gap.

    Dashed tape: dashes belonging to one line share ~the same x, so the biggest
    x-gap sits between the two lines. Either group may end up empty (single line).
    """
    if len(comps) == 1:
        cx = comps[0][0]
        return ([comps[0]], []) if cx < w / 2.0 else ([], [comps[0]])

    best_gap, best_i = -1.0, 0
    for i in range(len(comps) - 1):
        gap = comps[i + 1][0] - comps[i][0]
        if gap > best_gap:
            best_gap, best_i = gap, i

    if best_gap < MIN_SPLIT_GAP_FRAC * w:
        mean_x = sum(c[0] for c in comps) / len(comps)
        return (comps, []) if mean_x < w / 2.0 else ([], comps)

    return comps[: best_i + 1], comps[best_i + 1:]


def _blank(w):
    return {
        "left_x": None, "right_x": None, "lane_center": None, "lane_width": None,
        "confidence": "NONE", "n_pixels": 0, "left_line": None, "right_line": None,
        "both": False,
    }


def extract_lane_lines(mask, last_width, lane_width_cal):
    """Find lane geometry from a binary mask (uint8 0/255), in mask space."""
    h, w = mask.shape[:2]
    res = _blank(w)

    y0 = int(h * ROI_TOP_FRAC)
    y1 = int(h * ROI_BOTTOM_FRAC)
    if y1 <= y0:
        return res

    roi = np.zeros_like(mask)
    roi[y0:y1, :] = mask[y0:y1, :]
    n_pixels = int(cv2.countNonZero(roi))
    res["n_pixels"] = n_pixels
    if n_pixels < MIN_LANE_PIXELS:
        return res

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(roi, connectivity=8)
    comps = []
    for lbl in range(1, num):
        if stats[lbl, cv2.CC_STAT_AREA] < MIN_BLOB_AREA:
            continue
        comps.append((float(centroids[lbl][0]), lbl, int(stats[lbl, cv2.CC_STAT_AREA])))
    if not comps:
        res["confidence"] = "LOW"
        return res
    comps.sort(key=lambda c: c[0])

    left_comps, right_comps = _split_left_right(comps, w)
    left_line, n_left = _fit_line(labels, left_comps)
    right_line, n_right = _fit_line(labels, right_comps)

    y_bottom = y1 - 1
    left_x = _line_x_at(left_line, y_bottom) if left_line else None
    right_x = _line_x_at(right_line, y_bottom) if right_line else None

    width_min = LANE_WIDTH_MIN_FRAC * w
    width_max = LANE_WIDTH_MAX_FRAC * w
    fw = last_width or lane_width_cal or (0.4 * w)

    # Both edges found AND a plausible width -> trust it.
    if left_x is not None and right_x is not None and (right_x - left_x) >= width_min:
        lane_width = right_x - left_x
        conf = "HIGH" if lane_width <= width_max else "MEDIUM"
        res.update(
            left_x=left_x, right_x=right_x, lane_center=(left_x + right_x) / 2.0,
            lane_width=lane_width, confidence=conf, both=True,
            left_line=left_line, right_line=right_line,
        )
        return res

    # Otherwise treat as a single line (covers a bad split that landed inside one line).
    # Use whichever side had more pixels as the real line; estimate the partner.
    if left_x is not None or right_x is not None:
        if left_x is not None and (right_x is None or n_left >= n_right):
            res.update(
                left_x=left_x, right_x=left_x + fw, lane_center=left_x + fw / 2.0,
                lane_width=fw, confidence="MEDIUM", left_line=left_line,
            )
        else:
            res.update(
                left_x=right_x - fw, right_x=right_x, lane_center=right_x - fw / 2.0,
                lane_width=fw, confidence="MEDIUM", right_line=right_line,
            )
        return res

    res["confidence"] = "LOW"
    return res


def hsv_lane_mask(frame_bgr, target_shape):
    """Threshold a BGR frame for yellow tape and return a clean binary mask."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    th, tw = target_shape[:2]
    if (m.shape[0], m.shape[1]) != (th, tw):
        m = cv2.resize(m, (tw, th), interpolation=cv2.INTER_NEAREST)
    return m


# ══════════════════════════════════════════════════════════════════════════════
# Deviation estimator
# ══════════════════════════════════════════════════════════════════════════════
class LaneDeviation:
    """Fuses YOLO and HSV lane masks into a smoothed [0, 1] deviation score."""

    def __init__(self, ref_center=None, lane_width_cal=None, alpha=SMOOTH_ALPHA, mode="both"):
        self.ref_center = ref_center
        self.lane_width_cal = lane_width_cal
        self.last_width = lane_width_cal
        self.alpha = alpha
        self.mode = mode
        self.dev_smooth = 0.0

    def _ref(self, w):
        return self.ref_center if self.ref_center is not None else w / 2.0

    def update(self, yolo_mask, frame_bgr):
        w = yolo_mask.shape[1]
        ref = self._ref(w)
        snap = {
            "ref": ref, "mode": self.mode, "source": "NONE", "confidence": "NONE",
            "deviation": 0.0, "signed_offset": 0.0, "left_x": None, "right_x": None,
            "lane_center": None, "lane_width": None, "geom": None,
            "yolo_mask": yolo_mask, "hsv_mask": None, "src_mask": None,
        }

        info, source, src_mask = _blank(w), "NONE", None

        if self.mode in ("yolo", "both"):
            info = extract_lane_lines(yolo_mask, self.last_width, self.lane_width_cal)
            if info["confidence"] != "NONE":
                source, src_mask = "YOLO", yolo_mask

        if self.mode == "hsv" or (self.mode == "both" and info["confidence"] != "HIGH"):
            hmask = hsv_lane_mask(frame_bgr, yolo_mask.shape)
            snap["hsv_mask"] = hmask
            hinfo = extract_lane_lines(hmask, self.last_width, self.lane_width_cal)
            if self.mode == "hsv" or CONF_RANK[hinfo["confidence"]] > CONF_RANK[info["confidence"]]:
                if hinfo["confidence"] != "NONE":
                    info, source, src_mask = hinfo, "HSV", hmask

        # No usable detection -> treat as maximum deviation (fail-safe, not fail-silent).
        if info["confidence"] == "NONE":
            self.dev_smooth = 1.0
            snap["deviation"] = 1.0
            return snap

        self.last_width = info["lane_width"]
        offset = info["lane_center"] - ref
        half = max(info["lane_width"] / 2.0, 1.0)
        raw = min(abs(offset) / half, 1.0)
        if raw < DEADZONE:
            raw = 0.0
        self.dev_smooth = self.alpha * raw + (1.0 - self.alpha) * self.dev_smooth

        snap.update(
            source=source, confidence=info["confidence"], deviation=self.dev_smooth,
            signed_offset=offset, left_x=info["left_x"], right_x=info["right_x"],
            lane_center=info["lane_center"], lane_width=info["lane_width"],
            geom=info, src_mask=src_mask,
        )
        return snap


# ══════════════════════════════════════════════════════════════════════════════
# Drawing
# ══════════════════════════════════════════════════════════════════════════════
def overlay_mask(im0, mask, color, alpha=0.5):
    """Blend a mask-space binary mask onto the full-res frame in a solid color."""
    if mask is None:
        return im0
    big = cv2.resize(mask, (im0.shape[1], im0.shape[0]), interpolation=cv2.INTER_NEAREST)
    big = cv2.dilate(big, np.ones((5, 5), np.uint8), iterations=1)
    layer = np.zeros_like(im0)
    layer[big > 0] = color
    return cv2.addWeighted(im0, 1.0, layer, alpha, 0)


def draw_geometry(im0, snap, mask_shape):
    """Draw the fitted lane lines, lane center, reference line, and offset arrow."""
    g = snap.get("geom")
    if not g:
        return
    H, W = im0.shape[:2]
    mh, mw = mask_shape[:2]
    sx, sy = W / mw, H / mh
    y0 = int(ROI_TOP_FRAC * mh)
    y1 = int(ROI_BOTTOM_FRAC * mh) - 1

    def P(x, y):
        return int(x * sx), int(y * sy)

    if g.get("left_line"):
        cv2.line(im0, P(_line_x_at(g["left_line"], y0), y0),
                  P(_line_x_at(g["left_line"], y1), y1), (255, 0, 0), 4)
    if g.get("right_line"):
        cv2.line(im0, P(_line_x_at(g["right_line"], y0), y0),
                  P(_line_x_at(g["right_line"], y1), y1), (0, 0, 255), 4)

    lc, ref = g["lane_center"], snap["ref"]
    cv2.line(im0, P(lc, y0), P(lc, y1), (0, 255, 0), 2)
    cv2.line(im0, (int(ref * sx), 0), (int(ref * sx), H), (255, 255, 255), 2)
    ya = int((y0 + y1) / 2)
    cv2.arrowedLine(im0, P(ref, ya), P(lc, ya), (0, 255, 255), 3, tipLength=0.3)


def draw_hud(canvas, snap, fps):
    """Draw the on-screen FPS, deviation bar, source/confidence, and edge readout."""
    dev, src, conf = snap["deviation"], snap["source"], snap["confidence"]
    cv2.putText(canvas, f"FPS: {fps:.1f}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)

    dev_color = (0, 255, 0) if dev < 0.3 else (0, 165, 255) if dev < 0.6 else (0, 0, 255)
    cv2.putText(canvas, f"Deviation: {dev:.3f}", (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, dev_color, 3)

    bar_len = 400
    filled = int(min(max(dev, 0.0), 1.0) * bar_len)
    cv2.rectangle(canvas, (20, 130), (20 + bar_len, 160), (50, 50, 50), -1)
    cv2.rectangle(canvas, (20, 130), (20 + filled, 160), dev_color, -1)

    src_color = (0, 255, 0) if src == "YOLO" else (0, 165, 255) if src == "HSV" else (150, 150, 150)
    cv2.putText(canvas, f"mode={snap['mode']}  src={src}", (20, 205),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, src_color, 2)
    cv2.putText(canvas, f"Conf: {conf}", (20, 245),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    lx, rx = snap.get("left_x"), snap.get("right_x")
    cv2.putText(canvas, f"L={_f(lx)}  R={_f(rx)}  C={_f(snap.get('lane_center'))}",
                (20, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 255, 200), 2)


def _f(v):
    return f"{v:.1f}" if isinstance(v, (int, float)) else "--"


# ══════════════════════════════════════════════════════════════════════════════
# Args
# ══════════════════════════════════════════════════════════════════════════════
def make_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default="data/weights/yolopv2_full.engine")
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--nosave", action="store_true")
    p.add_argument("--project", default="runs/detect")
    p.add_argument("--name", default="exp_trt")
    p.add_argument("--exist-ok", action="store_true")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--mode", choices=["yolo", "hsv", "both"], default=DET_MODE)
    p.add_argument("--alpha", type=float, default=SMOOTH_ALPHA)
    p.add_argument("--ref-center", type=float, default=LANE_CENTER_CAL)
    p.add_argument("--lane-width", type=float, default=LANE_WIDTH_CAL)
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--calib-frames", type=int, default=60)
    p.add_argument("--print-every", type=int, default=5)
    p.add_argument("--max-frames", type=int, default=0)
    return p


# ══════════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════════
def detect(opt):
    save_img = not opt.nosave
    show_window = (not opt.headless) and (not opt.calibrate)

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    save_dir.mkdir(parents=True, exist_ok=True)

    engine = TRTEngine(opt.weights)
    stride = 32
    if USE_IMX_CAM:
        print("[CAMERA] Using IMX219 CSI camera")
        dataset = LoadCamera(img_size=TRT_W, stride=stride)
    else:
        dataset = LoadImages("/dev/video1", img_size=TRT_W, stride=stride)

    dummy = torch.zeros(1, 3, TRT_H, TRT_W, dtype=torch.float16, device="cuda")
    for _ in range(3):
        engine.infer(dummy)
    print("[TRT] Warmup done.")
    print(f"[MODE] detection mode = {opt.mode}   alpha = {opt.alpha}")
    if opt.ref_center is None:
        print("[CAL ] ref-center = IMAGE CENTER (not calibrated). "
              "Park centered, read C=, then pass --ref-center / --lane-width.")
    else:
        print(f"[CAL ] ref-center = {opt.ref_center}   lane-width = {opt.lane_width}")

    estimator = LaneDeviation(ref_center=opt.ref_center, lane_width_cal=opt.lane_width,
                               alpha=opt.alpha, mode=opt.mode)
    inf_time = AverageMeter()
    frame_count = 0
    t0 = time.time()
    calib_centers, calib_widths = [], []

    ts = time.strftime("%Y%m%d_%H%M%S")
    ann_path = str(save_dir / f"trt_{ts}_ann.mkv")
    raw_path = str(save_dir / f"trt_{ts}_raw.mkv")
    ann_writer = raw_writer = None
    writer_init = False

    try:
        for path, img, im0s, vid_cap in dataset:
            img_t = torch.from_numpy(img).to("cuda").half().unsqueeze(0) / 255.0
            if img_t.shape[2:] != (TRT_H, TRT_W):
                img_t = torch.nn.functional.interpolate(
                    img_t, size=(TRT_H, TRT_W), mode="bilinear", align_corners=False)

            t1 = time_synchronized()
            tensors = engine.infer(img_t)
            t2 = time_synchronized()
            inf_time.update(t2 - t1, 1)
            fps = 1.0 / (t2 - t1) if (t2 - t1) > 0 else 0.0

            da_seg = _get_tensor(tensors, DA_TENSOR_NAME)
            ll_seg = _get_tensor(tensors, LL_TENSOR_NAME)
            da_seg_mask = driving_area_mask(da_seg)
            ll_seg_mask = lane_line_mask(ll_seg)
            yolo_bin = (np.asarray(ll_seg_mask) > 0).astype(np.uint8) * 255

            snap = estimator.update(yolo_bin, im0s)
            latest_deviation = snap["deviation"]

            try:
                with open("/tmp/lane_deviation.txt", "w") as f:
                    f.write(str(latest_deviation))
            except OSError:
                pass

            # ---- diagnostic print (read L/R here to calibrate) ----
            if frame_count % max(opt.print_every, 1) == 0:
                print(f"mode={snap['mode']} src={snap['source']:<4} conf={snap['confidence']:<6} "
                      f"L={_f(snap['left_x'])} R={_f(snap['right_x'])} "
                      f"C={_f(snap['lane_center'])} W={_f(snap['lane_width'])} "
                      f"px={snap['yolo_mask'].any() and int((yolo_bin > 0).sum()) or 0} "
                      f"dev={latest_deviation:.3f} fps={fps:.1f}")

            # ---- calibration mode ----
            if opt.calibrate:
                if snap["geom"] and snap["geom"].get("both"):
                    calib_centers.append(snap["lane_center"])
                    calib_widths.append(snap["lane_width"])
                if len(calib_centers) >= opt.calib_frames:
                    ref = float(np.median(calib_centers))
                    wid = float(np.median(calib_widths))
                    print("\n================ CALIBRATION RESULT ================")
                    print(f"   --ref-center {ref:.1f}   --lane-width {wid:.1f}")
                    print("   (or set LANE_CENTER_CAL / LANE_WIDTH_CAL in the file)")
                    print("===================================================")
                    break
                if frame_count > opt.calib_frames * 30:
                    print("\n[CALIB] Could not gather enough clean both-line frames.")
                    if calib_centers:
                        print(f"   partial -> --ref-center {np.median(calib_centers):.1f} "
                              f"--lane-width {np.median(calib_widths):.1f}")
                    else:
                        print("   YOLO/HSV never saw both lanes. Try --mode hsv, move the track, "
                              "or lower MIN_LANE_PIXELS.")
                    break
                frame_count += 1
                continue

            # ---- visualization ----
            im0 = im0s.copy()
            show_seg_result(im0, (da_seg_mask, ll_seg_mask), is_demo=True)
            im0 = overlay_mask(im0, yolo_bin, (0, 255, 0), 0.45)               # YOLO lane = green
            if snap["source"] == "HSV":
                im0 = overlay_mask(im0, snap["hsv_mask"], (0, 140, 255), 0.45)  # HSV = orange
            draw_geometry(im0, snap, yolo_bin.shape)

            target_w, target_h = 960, 1080
            h, w = im0.shape[:2]
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            im0 = cv2.resize(im0, (new_w, new_h))
            top = (target_h - new_h) // 2
            bottom = target_h - new_h - top
            left = (target_w - new_w) // 2
            right = target_w - new_w - left
            im0 = cv2.copyMakeBorder(im0, top, bottom, left, right,
                                      cv2.BORDER_CONSTANT, value=(0, 0, 0))
            draw_hud(im0, snap, fps)

            if show_window:
                cv2.imshow("YOLOPv2-TRT Live (v2)", im0)

            if save_img:
                if not writer_init:
                    ann_writer = cv2.VideoWriter(ann_path, cv2.VideoWriter_fourcc(*"XVID"),
                                                  REC_FPS, (im0.shape[1], im0.shape[0]))
                    raw_writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"XVID"),
                                                  REC_FPS, (im0s.shape[1], im0s.shape[0]))
                    writer_init = True
                    print(f"[REC] Annotated -> {ann_path}")
                    print(f"[REC] Raw       -> {raw_path}")
                ann_writer.write(im0)
                raw_writer.write(im0s)

            frame_count += 1
            if show_window and (cv2.waitKey(1) & 0xFF == ord("q")):
                break
            if opt.max_frames and frame_count >= opt.max_frames:
                break

    except KeyboardInterrupt:
        print("\n[STOP] Interrupted by user.")
    finally:
        if inf_time.count > 0:
            print(f"\nAvg inference: {inf_time.avg:.4f}s/frame  ({1 / inf_time.avg:.1f} FPS)")
        print(f"Total: {time.time() - t0:.3f}s")
        if ann_writer:
            ann_writer.release()
        if raw_writer:
            raw_writer.release()
        if show_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    opt = make_parser().parse_args()
    print(opt)
    with torch.no_grad():
        detect(opt)
