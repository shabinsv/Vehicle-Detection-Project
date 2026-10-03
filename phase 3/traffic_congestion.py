"""Simple YOLO based traffic analysis for an Academy project."""

from collections import defaultdict, deque
from pathlib import Path
import math

import cv2
import numpy as np
from ultralytics import YOLO


# Change this value to the path of your traffic video.
VIDEO_PATH = "traffic.mp4"
MODEL_PATH = "yolo11n.pt"

# Approximate conversion only. Real-world speed needs camera calibration.
PIXEL_TO_METER = 0.05
SLOW_SPEED_THRESHOLD = 10.0  # km/h

# congestion thresholds.
LOW_MAX_VEHICLES = 10
MODERATE_MAX_VEHICLES = 20
HEAVY_MIN_VEHICLES = 21
LOW_OCCUPANCY_MAX = 15.0
MODERATE_OCCUPANCY_MAX = 35.0
HIGH_OCCUPANCY_MIN = 35.0
LOW_SPEED_MIN = 30.0
MODERATE_SPEED_MIN = 15.0
VERY_LOW_SPEED_MAX = 5.0
CONGESTED_MIN_SLOW_VEHICLES = 3

VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle"}
CLASS_NAMES = {"car": "Cars", "bus": "Buses", "truck": "Trucks", "motorcycle": "Motorcycles"}
HISTORY_LENGTH = 5


def classify_traffic(vehicle_count, average_speed, occupancy, slow_count):
    """Apply simple, editable rules to label the current traffic conditions."""
    if (
        average_speed is not None
        and average_speed <= VERY_LOW_SPEED_MAX
        and occupancy >= HIGH_OCCUPANCY_MIN
        and slow_count >= CONGESTED_MIN_SLOW_VEHICLES
    ):
        return "CONGESTED"

    if (
        vehicle_count >= HEAVY_MIN_VEHICLES
        and average_speed is not None
        and average_speed < MODERATE_SPEED_MIN
        and occupancy >= HIGH_OCCUPANCY_MIN
    ):
        return "HEAVY"

    if vehicle_count > LOW_MAX_VEHICLES or occupancy >= MODERATE_OCCUPANCY_MAX:
        return "MODERATE"

    if (
        average_speed is not None
        and average_speed >= LOW_SPEED_MIN
        and occupancy < LOW_OCCUPANCY_MAX
        and vehicle_count <= LOW_MAX_VEHICLES
    ):
        return "LOW"

    # With insufficient speed data, use count and occupancy rather than inventing a speed.
    if vehicle_count <= LOW_MAX_VEHICLES and occupancy < LOW_OCCUPANCY_MAX:
        return "LOW"
    if vehicle_count <= MODERATE_MAX_VEHICLES:
        return "MODERATE"
    return "HEAVY"


def main():
    video_file = Path(VIDEO_PATH)
    if not video_file.is_file():
        print(f"Video not found: {video_file.resolve()}")
        print("Set VIDEO_PATH near the top of traffic_congestion.py to your video file.")
        return
    model = YOLO(MODEL_PATH)
    capture = cv2.VideoCapture(str(video_file))
    if not capture.isOpened():
        print(f"Could not open video: {video_file.resolve()}")
        return

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        fps = None

    # Persist unique IDs and their last few center points across video frames.
    seen_ids = set()
    id_to_class = {}
    point_history = defaultdict(lambda: deque(maxlen=HISTORY_LENGTH))
    speed_by_id = {}

    class_id_to_name = {index: name for index, name in model.names.items()}
    vehicle_class_ids = [
        index for index, name in class_id_to_name.items() if name in VEHICLE_CLASSES
    ]

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        frame_height, frame_width = frame.shape[:2]
        frame_area = frame_width * frame_height
        result = model.track(
            frame,
            persist=True,
            classes=vehicle_class_ids,
            tracker="bytetrack.yaml",
            verbose=False,
        )[0]

        current_boxes = []
        current_vehicle_area = 0.0
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            coordinates = boxes.xyxy.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)
            track_ids = boxes.id.cpu().numpy().astype(int)

            for coordinates_one, class_id, track_id in zip(coordinates, class_ids, track_ids):
                vehicle_type = class_id_to_name.get(class_id, "vehicle")
                if vehicle_type not in VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = map(int, coordinates_one)
                x1, x2 = max(0, x1), min(frame_width, x2)
                y1, y2 = max(0, y1), min(frame_height, y2)
                box_area = max(0, x2 - x1) * max(0, y2 - y1)
                current_vehicle_area += box_area
                current_boxes.append((x1, y1, x2, y2, track_id, vehicle_type))

                seen_ids.add(track_id)
                id_to_class[track_id] = vehicle_type
                center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                point_history[track_id].append(center)

                # Estimate speed from center displacement over elapsed video time.
                # Require several observations and valid FPS before showing a value.
                points = point_history[track_id]
                if fps is not None and len(points) >= HISTORY_LENGTH:
                    pixel_distance = math.dist(points[0], points[-1])
                    elapsed_seconds = (len(points) - 1) / fps
                    if elapsed_seconds > 0:
                        speed_kmh = (pixel_distance * PIXEL_TO_METER / elapsed_seconds) * 3.6
                        if math.isfinite(speed_kmh):
                            speed_by_id[track_id] = speed_kmh

        # Bounding-box area is only an approximate occupancy measure, not a road mask.
        occupancy = min(100.0, (current_vehicle_area / frame_area) * 100.0) if frame_area else 0.0

        known_speeds = [speed_by_id[track_id] for track_id in seen_ids if track_id in speed_by_id]
        average_speed = sum(known_speeds) / len(known_speeds) if known_speeds else None
        slow_count = sum(speed < SLOW_SPEED_THRESHOLD for speed in known_speeds)
        traffic_status = classify_traffic(len(seen_ids), average_speed, occupancy, slow_count)

        for x1, y1, x2, y2, track_id, vehicle_type in current_boxes:
            color = (0, 220, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{vehicle_type.title()} ID: {track_id}"
            cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        class_counts = {name: 0 for name in CLASS_NAMES}
        for vehicle_type in id_to_class.values():
            if vehicle_type in class_counts:
                class_counts[vehicle_type] += 1

        speed_text = f"{average_speed:.1f} km/h" if average_speed is not None else "N/A"
        overlay_lines = [
            "Traffic Analysis",
            f"Total Vehicles: {len(seen_ids)}",
            f"Cars: {class_counts['car']}  Buses: {class_counts['bus']}",
            f"Trucks: {class_counts['truck']}  Motorcycles: {class_counts['motorcycle']}",
            f"Estimated Average Speed: {speed_text}",
            f"Occupancy: {occupancy:.1f}%",
            f"Slow Vehicles: {slow_count}",
            f"Traffic Status: {traffic_status}",
        ]
        cv2.rectangle(frame, (8, 8), (440, 212), (0, 0, 0), -1)
        for line_number, line in enumerate(overlay_lines):
            cv2.putText(
                frame,
                line,
                (18, 34 + line_number * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.imshow("AI Road Traffic Congestion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
