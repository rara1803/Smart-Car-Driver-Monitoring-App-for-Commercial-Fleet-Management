#!/usr/bin/env python3
"""
carla_manual_validation.py
===========================
Manual (keyboard-driven) CARLA validation harness for the lane-deviation pipeline.

Spawns a vehicle in CARLA, drives it manually via arrow keys, runs YOLOPv2 on the
front camera every N frames to estimate lane deviation, detects front obstacles
(actor-based + a static-object sensor fallback), and logs everything to a JSONL
file. Optionally streams the same telemetry to the Jetson over WebSocket.

Controls:
    UP / DOWN     forward / reverse
    LEFT / RIGHT  steer
    SPACE         brake
    Q             quit

Usage:
    python3 carla_manual_validation.py
"""

import sys
import time
import json
import queue
import math
from pathlib import Path

import carla
import keyboard
import numpy as np
import torch
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from utils.utils import select_device, lane_line_mask


# ============================================================
# PATHS
# ============================================================
WEIGHTS_PATH = PROJECT_ROOT / "models" / "yolopv2.pt"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "manual_live_lane_deviation_hybrid_obstacle_log.jsonl"
DEBUG_DIR = PROJECT_ROOT / "outputs" / "debug_live"

DEBUG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================
IMG_SIZE = 640

# Higher value = smoother CARLA, but slower AI updates.
INFERENCE_EVERY_N_FRAMES = 8

# Save debug camera + lane mask every N frames.
DEBUG_SAVE_EVERY_N_FRAMES = 80

# Vehicle control
STEER_RATE = 0.035
FORWARD_THROTTLE = 0.35
REVERSE_THROTTLE = 0.30
MAX_STEER = 0.45

# Actor-based obstacle detection (detects real spawned vehicles/pedestrians)
FRONT_OBSTACLE_DISTANCE_M = 40.0
FRONT_OBSTACLE_HALF_WIDTH_M = 8.0

# Static obstacle fallback sensor.
# Keep this narrow so it doesn't detect every sign/light/pole.
STATIC_OBSTACLE_SENSOR_DISTANCE = 12.0
STATIC_OBSTACLE_SENSOR_HIT_RADIUS = 0.7
STATIC_OBSTACLE_TIMEOUT_SECONDS = 0.25

# Jetson WebSocket telemetry.
# Set True + point JETSON_WS_URL at your Jetson's address to stream live to hardware.
ENABLE_JETSON_WEBSOCKET = False
JETSON_WS_URL = "ws://<jetson-ip>:8765"

ws = None


# ============================================================
# WEBSOCKET TELEMETRY SENDER
# ============================================================
def connect_to_jetson_websocket():
    global ws

    if not ENABLE_JETSON_WEBSOCKET:
        return

    try:
        from websocket import create_connection

        ws = create_connection(JETSON_WS_URL, timeout=3)
        print("Connected to Jetson WebSocket:", JETSON_WS_URL)

    except Exception as e:
        ws = None
        print("Could not connect to Jetson WebSocket:", e)


def send_telemetry_to_jetson(record):
    global ws

    if not ENABLE_JETSON_WEBSOCKET:
        return

    if ws is None:
        connect_to_jetson_websocket()

    if ws is not None:
        try:
            ws.send(json.dumps(record))
        except Exception as e:
            print("WebSocket send failed:", e)
            ws = None


# ============================================================
# LANE DEVIATION ESTIMATOR
#
# Output:
#   lane_deviation: float in [0.0, 1.0]
#
# Behavior:
#   clear lane pair detected      -> compute normally
#   no lane pixels at all         -> 0.0
#   lane pixels exist but unclear -> hold previous value
# ============================================================
class LaneDeviationEstimator:
    """Histogram-based left/right lane pairing and normalized deviation scoring."""

    def __init__(self):
        self.last_deviation = 0.0
        self.last_left_x = None
        self.last_right_x = None
        self.last_lane_width = None

    def estimate(self, lane_mask):
        h, w = lane_mask.shape
        vehicle_center = w / 2.0

        roi_top = int(h * 0.50)
        roi_bottom = int(h * 0.90)
        roi = lane_mask[roi_top:roi_bottom, :]

        ys, xs = np.where(roi > 0)
        if len(xs) < 15:
            return self._no_lane()

        hist = np.bincount(xs, minlength=w).astype(np.float32)

        kernel_size = 21
        kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
        hist_smooth = np.convolve(hist, kernel, mode="same")

        max_value = hist_smooth.max()
        if max_value <= 0:
            return self._no_lane()

        threshold = max_value * 0.25
        peak_pixels = np.where(hist_smooth > threshold)[0]
        if len(peak_pixels) == 0:
            return self._unclear_lane()

        groups = [[peak_pixels[0]]]
        for x in peak_pixels[1:]:
            if x - groups[-1][-1] <= 16:
                groups[-1].append(x)
            else:
                groups.append([x])

        candidates = [float(np.mean(g)) for g in groups]
        left_candidates = [x for x in candidates if x < vehicle_center]
        right_candidates = [x for x in candidates if x > vehicle_center]

        if not left_candidates or not right_candidates:
            return self._unclear_lane()

        pairs = []
        for left_x in left_candidates:
            for right_x in right_candidates:
                lane_width = right_x - left_x
                if lane_width < 80 or lane_width > 560:
                    continue

                left_support = np.abs(xs - left_x) < 25
                right_support = np.abs(xs - right_x) < 25
                left_count = int(np.sum(left_support))
                right_count = int(np.sum(right_support))
                if left_count < 10 or right_count < 10:
                    continue

                left_y_spread = np.ptp(ys[left_support]) if left_count > 0 else 0
                right_y_spread = np.ptp(ys[right_support]) if right_count > 0 else 0
                if left_y_spread < 15 or right_y_spread < 15:
                    continue

                lane_center = (left_x + right_x) / 2.0
                pair_score = abs(vehicle_center - lane_center)

                if self.last_left_x is not None and self.last_right_x is not None:
                    continuity_error = abs(left_x - self.last_left_x) + abs(right_x - self.last_right_x)
                    pair_score += 0.25 * continuity_error

                pairs.append((pair_score, left_x, right_x, lane_width, lane_center))

        if not pairs:
            return self._unclear_lane()

        pairs.sort(key=lambda p: p[0])
        _, left_x, right_x, lane_width, lane_center = pairs[0]

        raw_deviation = abs(vehicle_center - lane_center) / (lane_width / 2.0)
        raw_deviation = float(min(raw_deviation, 1.0))
        if raw_deviation < 0.03:
            raw_deviation = 0.0

        alpha = 0.35
        smoothed = alpha * raw_deviation + (1.0 - alpha) * self.last_deviation

        self.last_deviation = smoothed
        self.last_left_x = left_x
        self.last_right_x = right_x
        self.last_lane_width = lane_width

        return self.last_deviation

    def _no_lane(self):
        self.last_deviation = 0.0
        self.last_left_x = None
        self.last_right_x = None
        self.last_lane_width = None
        return 0.0

    def _unclear_lane(self):
        return self.last_deviation


# ============================================================
# KEYBOARD CONTROL
# ============================================================
current_steer = 0.0


def read_keyboard_control():
    global current_steer

    forward = keyboard.is_pressed("up")
    reverse_pressed = keyboard.is_pressed("down")
    brake_pressed = keyboard.is_pressed("space")

    throttle = 0.0
    brake = 0.0
    reverse_mode = False

    if forward:
        throttle = FORWARD_THROTTLE
        reverse_mode = False
    elif reverse_pressed:
        throttle = REVERSE_THROTTLE
        reverse_mode = True

    if brake_pressed:
        throttle = 0.0
        brake = 1.0

    target_steer = 0.0
    if keyboard.is_pressed("left"):
        target_steer = -MAX_STEER
    if keyboard.is_pressed("right"):
        target_steer = MAX_STEER

    if target_steer > current_steer:
        current_steer = min(current_steer + STEER_RATE, target_steer)
    elif target_steer < current_steer:
        current_steer = max(current_steer - STEER_RATE, target_steer)
    else:
        if current_steer > 0:
            current_steer = max(0.0, current_steer - STEER_RATE)
        elif current_steer < 0:
            current_steer = min(0.0, current_steer + STEER_RATE)

    return carla.VehicleControl(
        throttle=throttle,
        steer=current_steer,
        brake=brake,
        reverse=reverse_mode,
        hand_brake=False,
    )


def movement_key_pressed():
    return keyboard.is_pressed("up") or keyboard.is_pressed("down")


# ============================================================
# VEHICLE STATE HELPERS
# ============================================================
def get_longitudinal_acceleration(vehicle):
    acceleration = vehicle.get_acceleration()
    forward_vector = vehicle.get_transform().get_forward_vector()
    return float(
        acceleration.x * forward_vector.x
        + acceleration.y * forward_vector.y
        + acceleration.z * forward_vector.z
    )


def get_speed(vehicle):
    velocity = vehicle.get_velocity()
    speed_mps = (velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5
    return float(speed_mps * 3.6)


# ============================================================
# ACTOR-BASED FRONT OBSTACLE CHECK
#
# Detects real spawned vehicle.* and walker.pedestrian.* actors. Handles
# moving/spawned traffic cleanly.
# ============================================================
def get_front_actor_obstacle(world, ego_vehicle):
    ego_transform = ego_vehicle.get_transform()
    ego_location = ego_transform.location
    ego_forward = ego_transform.get_forward_vector()

    yaw_rad = math.radians(ego_transform.rotation.yaw)
    right_x = math.cos(yaw_rad + math.pi / 2.0)
    right_y = math.sin(yaw_rad + math.pi / 2.0)

    closest_actor = None
    closest_forward_distance = None

    actors = list(world.get_actors().filter("vehicle.*")) + \
        list(world.get_actors().filter("walker.pedestrian.*"))

    for actor in actors:
        if actor.id == ego_vehicle.id:
            continue

        actor_location = actor.get_location()
        dx = actor_location.x - ego_location.x
        dy = actor_location.y - ego_location.y

        forward_distance = dx * ego_forward.x + dy * ego_forward.y
        lateral_distance = abs(dx * right_x + dy * right_y)

        if forward_distance <= 1.0 or forward_distance > FRONT_OBSTACLE_DISTANCE_M:
            continue
        if lateral_distance > FRONT_OBSTACLE_HALF_WIDTH_M:
            continue

        if closest_forward_distance is None or forward_distance < closest_forward_distance:
            closest_forward_distance = forward_distance
            closest_actor = actor

    if closest_actor is None:
        return {"detected": False, "distance_m": None, "type": None}

    return {"detected": True, "distance_m": float(closest_forward_distance), "type": closest_actor.type_id}


# ============================================================
# STATIC OBSTACLE SENSOR FALLBACK
#
# Only for parked/static car-like objects. Ignores traffic lights, signs,
# poles, street furniture, roads, buildings, walls, fences, vegetation, etc.
# ============================================================
latest_static_obstacle = {"detected": False, "distance_m": None, "last_seen_time": 0.0}

STATIC_IGNORE_KEYWORDS = (
    "traffic", "light", "sign", "pole", "street", "road", "sidewalk",
    "building", "wall", "fence", "vegetation", "terrain", "ground", "lamp",
    "post", "tree",
)
STATIC_ACCEPT_KEYWORDS = ("vehicle", "car", "truck", "van", "bus")


def static_obstacle_callback(event):
    actor_type = event.other_actor.type_id.lower()

    if any(word in actor_type for word in STATIC_IGNORE_KEYWORDS):
        return
    if not any(word in actor_type for word in STATIC_ACCEPT_KEYWORDS):
        return

    latest_static_obstacle["detected"] = True
    latest_static_obstacle["distance_m"] = float(event.distance)
    latest_static_obstacle["last_seen_time"] = time.time()


def get_static_obstacle_fallback():
    current_time = time.time()

    if latest_static_obstacle["detected"]:
        if current_time - latest_static_obstacle["last_seen_time"] <= STATIC_OBSTACLE_TIMEOUT_SECONDS:
            return {
                "detected": True,
                "distance_m": latest_static_obstacle["distance_m"],
                "type": "static_obstacle",
            }

    latest_static_obstacle["detected"] = False
    latest_static_obstacle["distance_m"] = None
    return {"detected": False, "distance_m": None, "type": None}


def get_hybrid_front_obstacle(world, ego_vehicle):
    """Prefer real actor detection; fall back to the static-object sensor."""
    actor_obstacle = get_front_actor_obstacle(world, ego_vehicle)
    if actor_obstacle["detected"]:
        return actor_obstacle
    return get_static_obstacle_fallback()


# ============================================================
# LOAD YOLOPV2
# ============================================================
device = select_device("0")

model = torch.jit.load(str(WEIGHTS_PATH))
model = model.to(device)
model.eval()

half = device.type != "cpu"
if half:
    model.half()

print("Model loaded on:", device)

if device.type != "cpu":
    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(device).type_as(next(model.parameters()))
    with torch.no_grad():
        model(dummy)


# ============================================================
# CONNECT TO CARLA
# ============================================================
client = carla.Client("127.0.0.1", 2000)
client.set_timeout(60.0)

world = client.get_world()
bp_lib = world.get_blueprint_library()

original_settings = world.get_settings()
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

vehicle = None
camera = None
static_obstacle_sensor = None
image_queue = queue.Queue(maxsize=5)

lane_estimator = LaneDeviationEstimator()


try:
    vehicle_bp = bp_lib.filter("vehicle.dodge.charger_2020")[0]
    spawn_point = world.get_map().get_spawn_points()[20]

    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print("Spawned:", vehicle.type_id)

    # ---- RGB camera ----
    camera_bp = bp_lib.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", "640")
    camera_bp.set_attribute("image_size_y", "384")
    camera_bp.set_attribute("fov", "90")

    camera_transform = carla.Transform(carla.Location(x=0.6, z=1.35), carla.Rotation(pitch=-12))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    def camera_callback(image):
        if image_queue.full():
            try:
                image_queue.get_nowait()
            except queue.Empty:
                pass
        image_queue.put(image)

    camera.listen(camera_callback)

    # ---- Static obstacle sensor fallback ----
    # only_dynamics=false so it can also see static map objects; the callback
    # filters by type so it doesn't fire on every sign/pole/building.
    static_obstacle_bp = bp_lib.find("sensor.other.obstacle")
    static_obstacle_bp.set_attribute("distance", str(STATIC_OBSTACLE_SENSOR_DISTANCE))
    static_obstacle_bp.set_attribute("hit_radius", str(STATIC_OBSTACLE_SENSOR_HIT_RADIUS))
    static_obstacle_bp.set_attribute("only_dynamics", "false")

    static_obstacle_transform = carla.Transform(
        carla.Location(x=3.0, z=1.2), carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
    )
    static_obstacle_sensor = world.spawn_actor(
        static_obstacle_bp, static_obstacle_transform, attach_to=vehicle
    )
    static_obstacle_sensor.listen(static_obstacle_callback)

    spectator = world.get_spectator()

    print("Manual live lane deviation with hybrid obstacle detection started.")
    print("Controls: UP=forward  DOWN=reverse  LEFT/RIGHT=steer  SPACE=brake  Q=quit")
    print("Saving to:", OUTPUT_PATH)
    print("Actor obstacle distance:", FRONT_OBSTACLE_DISTANCE_M, "meters")
    print("Actor obstacle half width:", FRONT_OBSTACLE_HALF_WIDTH_M, "meters")
    print("Static obstacle fallback distance:", STATIC_OBSTACLE_SENSOR_DISTANCE, "meters")

    if ENABLE_JETSON_WEBSOCKET:
        print("Sending live telemetry to Jetson:", JETSON_WS_URL)
    else:
        print("Jetson WebSocket disabled. Data will only be printed and saved locally.")

    start_time = time.time()
    frame_id = 0
    latest_deviation = 0.0

    # Force lane deviation to zero until the driver first presses UP/DOWN,
    # so the score doesn't reflect a stationary/unstarted vehicle.
    vehicle_has_started = False

    with open(OUTPUT_PATH, "w") as f:
        while True:
            if keyboard.is_pressed("q"):
                print("Q pressed. Stopping.")
                break

            if movement_key_pressed():
                vehicle_has_started = True

            vehicle.apply_control(read_keyboard_control())
            world.tick()

            try:
                image = image_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if frame_id % INFERENCE_EVERY_N_FRAMES == 0:
                arr = np.frombuffer(image.raw_data, dtype=np.uint8)
                arr = arr.reshape((image.height, image.width, 4))

                frame_bgr = arr[:, :, :3]
                frame_rgb = frame_bgr[:, :, ::-1]

                img = np.ascontiguousarray(frame_rgb.transpose(2, 0, 1))
                img_tensor = torch.from_numpy(img).to(device)
                img_tensor = img_tensor.half() if half else img_tensor.float()
                img_tensor /= 255.0
                img_tensor = img_tensor.unsqueeze(0)

                with torch.no_grad():
                    [pred, anchor_grid], seg, ll = model(img_tensor)

                ll_mask = lane_line_mask(ll)
                latest_deviation = lane_estimator.estimate(ll_mask)

                if not vehicle_has_started:
                    latest_deviation = 0.0
                    lane_estimator.last_deviation = 0.0

                if frame_id % DEBUG_SAVE_EVERY_N_FRAMES == 0:
                    camera_path = DEBUG_DIR / f"camera_frame_{frame_id:05d}.png"
                    mask_path = DEBUG_DIR / f"lane_mask_{frame_id:05d}.png"
                    cv2.imwrite(str(camera_path), frame_bgr)
                    cv2.imwrite(str(mask_path), ll_mask * 255)

            acceleration = get_longitudinal_acceleration(vehicle)
            speed_kmh = get_speed(vehicle)
            front_obstacle = get_hybrid_front_obstacle(world, vehicle)

            record = {
                "timestamp": round(time.time() - start_time, 3),
                "lane_deviation": round(float(latest_deviation), 3),
                "acceleration_mps2": round(acceleration, 3),
                "speed_kmh": round(speed_kmh, 3),
                "obstacle_detected": bool(front_obstacle["detected"]),
                "obstacle_distance_m": (
                    None if front_obstacle["distance_m"] is None
                    else round(float(front_obstacle["distance_m"]), 3)
                ),
                "obstacle_type": front_obstacle["type"],
            }

            f.write(json.dumps(record) + "\n")
            f.flush()

            send_telemetry_to_jetson(record)

            if frame_id % 5 == 0:
                print(record)

            transform = vehicle.get_transform()
            fwd = transform.get_forward_vector()
            spectator.set_transform(carla.Transform(
                transform.location + carla.Location(x=-8 * fwd.x, y=-8 * fwd.y, z=5),
                carla.Rotation(pitch=-25, yaw=transform.rotation.yaw),
            ))

            frame_id += 1

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    if static_obstacle_sensor is not None:
        static_obstacle_sensor.stop()
        static_obstacle_sensor.destroy()

    if camera is not None:
        camera.stop()
        camera.destroy()

    if vehicle is not None:
        vehicle.destroy()

    if ws is not None:
        try:
            ws.close()
        except Exception:
            pass

    world.apply_settings(original_settings)
    print("Cleaned and restored CARLA settings.")
