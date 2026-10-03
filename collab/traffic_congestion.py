"""Live YOLO-based traffic analysis for Google Colab."""

from collections import defaultdict, deque
from pathlib import Path
import math
import time

import cv2
import numpy as np
from ultralytics import YOLO
from google.colab.patches import cv2_imshow
from IPython.display import clear_output


# ============================================================
# SETTINGS
# ============================================================

VIDEO_PATH = "/content/traffic.mp4"
MODEL_PATH = "yolo11n.pt"

# Approximate conversion only.
# Real-world speed requires camera calibration.
PIXEL_TO_METER = 0.05

SLOW_SPEED_THRESHOLD = 10.0  # km/h

# Traffic thresholds
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

# Vehicle classes detected by YOLO
VEHICLE_CLASSES = {
    "car",
    "bus",
    "truck",
    "motorcycle"
}

CLASS_NAMES = {
    "car": "Cars",
    "bus": "Buses",
    "truck": "Trucks",
    "motorcycle": "Motorcycles"
}

# Number of points used for speed estimation
HISTORY_LENGTH = 5

# Display every Nth frame.
# 1 = every frame
# 2 = every second frame
# 3 = every third frame
DISPLAY_EVERY_N_FRAMES = 3


# ============================================================
# TRAFFIC CLASSIFICATION
# ============================================================

def classify_traffic(
    vehicle_count,
    average_speed,
    occupancy,
    slow_count
):
    """Classify current traffic conditions."""

    # Strong congestion condition
    if (
        average_speed is not None
        and average_speed <= VERY_LOW_SPEED_MAX
        and occupancy >= HIGH_OCCUPANCY_MIN
        and slow_count >= CONGESTED_MIN_SLOW_VEHICLES
    ):
        return "CONGESTED"

    # Heavy traffic
    if (
        vehicle_count >= HEAVY_MIN_VEHICLES
        and average_speed is not None
        and average_speed < MODERATE_SPEED_MIN
        and occupancy >= HIGH_OCCUPANCY_MIN
    ):
        return "HEAVY"

    # Moderate traffic
    if (
        vehicle_count > LOW_MAX_VEHICLES
        or occupancy >= MODERATE_OCCUPANCY_MAX
    ):
        return "MODERATE"

    # Low traffic
    if (
        average_speed is not None
        and average_speed >= LOW_SPEED_MIN
        and occupancy < LOW_OCCUPANCY_MAX
        and vehicle_count <= LOW_MAX_VEHICLES
    ):
        return "LOW"

    # If speed isn't available yet
    if (
        vehicle_count <= LOW_MAX_VEHICLES
        and occupancy < LOW_OCCUPANCY_MAX
    ):
        return "LOW"

    if vehicle_count <= MODERATE_MAX_VEHICLES:
        return "MODERATE"

    return "HEAVY"


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # CHECK VIDEO
    # --------------------------------------------------------

    video_file = Path(VIDEO_PATH)

    if not video_file.is_file():

        print("❌ Video not found!")
        print(f"Expected location: {video_file}")
        print()
        print("Upload your traffic video to Colab first.")

        return

    print("✅ Video found!")
    print(f"Video: {video_file}")
    print()

    # --------------------------------------------------------
    # LOAD YOLO
    # --------------------------------------------------------

    print("Loading YOLO model...")

    model = YOLO(MODEL_PATH)

    print("✅ YOLO model loaded!")
    print()

    # --------------------------------------------------------
    # OPEN VIDEO
    # --------------------------------------------------------

    capture = cv2.VideoCapture(str(video_file))

    if not capture.isOpened():

        print("❌ Could not open video.")

        return

    # Video properties
    fps = capture.get(cv2.CAP_PROP_FPS)

    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0

    total_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    original_width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    original_height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    print(f"Resolution: {original_width} x {original_height}")
    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")
    print()

    # --------------------------------------------------------
    # TRACKING DATA
    # --------------------------------------------------------

    point_history = defaultdict(
        lambda: deque(maxlen=HISTORY_LENGTH)
    )

    speed_by_id = {}

    # Convert YOLO class IDs to class names
    class_id_to_name = {
        index: name
        for index, name in model.names.items()
    }

    # Find IDs corresponding to vehicles
    vehicle_class_ids = [
        index
        for index, name in class_id_to_name.items()
        if name in VEHICLE_CLASSES
    ]

    print("Vehicle classes:")
    print(vehicle_class_ids)
    print()

    # --------------------------------------------------------
    # FRAME COUNTER
    # --------------------------------------------------------

    frame_number = 0

    start_time = time.time()

    # ========================================================
    # PROCESS VIDEO
    # ========================================================

    while True:

        ok, frame = capture.read()

        if not ok:
            break

        frame_number += 1

        frame_height, frame_width = frame.shape[:2]

        frame_area = (
            frame_width *
            frame_height
        )

        # ----------------------------------------------------
        # YOLO TRACKING
        # ----------------------------------------------------

        result = model.track(
            frame,
            persist=True,
            classes=vehicle_class_ids,
            tracker="bytetrack.yaml",
            verbose=False
        )[0]

        # Store currently detected vehicles
        current_boxes = []

        # Total area occupied by vehicles
        current_vehicle_area = 0.0

        boxes = result.boxes

        if (
            boxes is not None
            and boxes.id is not None
        ):

            coordinates = (
                boxes.xyxy
                .cpu()
                .numpy()
            )

            class_ids = (
                boxes.cls
                .cpu()
                .numpy()
                .astype(int)
            )

            track_ids = (
                boxes.id
                .cpu()
                .numpy()
                .astype(int)
            )

            # ------------------------------------------------
            # PROCESS EACH VEHICLE
            # ------------------------------------------------

            for (
                coordinates_one,
                class_id,
                track_id
            ) in zip(
                coordinates,
                class_ids,
                track_ids
            ):

                vehicle_type = class_id_to_name.get(
                    class_id,
                    "vehicle"
                )

                # Only process vehicle classes
                if vehicle_type not in VEHICLE_CLASSES:
                    continue

                # Bounding box coordinates
                x1, y1, x2, y2 = map(
                    int,
                    coordinates_one
                )

                # Keep coordinates inside frame
                x1 = max(0, x1)
                x2 = min(frame_width, x2)

                y1 = max(0, y1)
                y2 = min(frame_height, y2)

                # Bounding box area
                box_area = (
                    max(0, x2 - x1)
                    *
                    max(0, y2 - y1)
                )

                current_vehicle_area += box_area

                # Save current vehicle
                current_boxes.append(
                    (
                        x1,
                        y1,
                        x2,
                        y2,
                        track_id,
                        vehicle_type
                    )
                )

                # ------------------------------------------------
                # VEHICLE CENTER
                # ------------------------------------------------

                center = (
                    (x1 + x2) / 2.0,
                    (y1 + y2) / 2.0
                )

                point_history[
                    track_id
                ].append(center)

                # ------------------------------------------------
                # SPEED ESTIMATION
                # ------------------------------------------------

                points = point_history[track_id]

                if (
                    fps is not None
                    and len(points) >= HISTORY_LENGTH
                ):

                    pixel_distance = math.dist(
                        points[0],
                        points[-1]
                    )

                    elapsed_seconds = (
                        len(points) - 1
                    ) / fps

                    if elapsed_seconds > 0:

                        speed_kmh = (
                            pixel_distance
                            *
                            PIXEL_TO_METER
                            /
                            elapsed_seconds
                        ) * 3.6

                        if math.isfinite(speed_kmh):

                            speed_by_id[
                                track_id
                            ] = speed_kmh

        # ====================================================
        # CURRENT VEHICLE COUNT
        # ====================================================

        current_vehicle_count = len(
            current_boxes
        )

        # ====================================================
        # OCCUPANCY
        # ====================================================

        if frame_area > 0:

            occupancy = min(
                100.0,
                (
                    current_vehicle_area
                    /
                    frame_area
                ) * 100.0
            )

        else:

            occupancy = 0.0

        # ====================================================
        # CURRENT VEHICLE SPEEDS
        # ====================================================

        current_track_ids = {
            box[4]
            for box in current_boxes
        }

        known_speeds = [
            speed_by_id[track_id]
            for track_id in current_track_ids
            if track_id in speed_by_id
        ]

        if known_speeds:

            average_speed = (
                sum(known_speeds)
                /
                len(known_speeds)
            )

        else:

            average_speed = None

        # Count slow vehicles
        slow_count = sum(
            speed < SLOW_SPEED_THRESHOLD
            for speed in known_speeds
        )

        # ====================================================
        # VEHICLE CLASS COUNTS
        # ====================================================

        class_counts = {
            name: 0
            for name in CLASS_NAMES
        }

        for box in current_boxes:

            vehicle_type = box[5]

            if vehicle_type in class_counts:

                class_counts[
                    vehicle_type
                ] += 1

        # ====================================================
        # TRAFFIC STATUS
        # ====================================================

        traffic_status = classify_traffic(
            current_vehicle_count,
            average_speed,
            occupancy,
            slow_count
        )

        # ====================================================
        # DRAW VEHICLE BOXES
        # ====================================================

        for (
            x1,
            y1,
            x2,
            y2,
            track_id,
            vehicle_type
        ) in current_boxes:

            # Green bounding box
            color = (0, 220, 0)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # Vehicle label
            label = (
                f"{vehicle_type.title()} "
                f"ID: {track_id}"
            )

            cv2.putText(
                frame,
                label,
                (
                    x1,
                    max(20, y1 - 8)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )

        # ====================================================
        # SPEED TEXT
        # ====================================================

        if average_speed is not None:

            speed_text = (
                f"{average_speed:.1f} km/h"
            )

        else:

            speed_text = "N/A"

        # ====================================================
        # INFORMATION PANEL
        # ====================================================

        overlay_lines = [

            "AI TRAFFIC ANALYSIS",

            f"Vehicles: "
            f"{current_vehicle_count}",

            f"Cars: "
            f"{class_counts['car']}   "
            f"Buses: "
            f"{class_counts['bus']}",

            f"Trucks: "
            f"{class_counts['truck']}   "
            f"Motorcycles: "
            f"{class_counts['motorcycle']}",

            f"Average Speed: "
            f"{speed_text}",

            f"Occupancy: "
            f"{occupancy:.1f}%",

            f"Slow Vehicles: "
            f"{slow_count}",

            f"Traffic Status: "
            f"{traffic_status}"
        ]

        # Panel height
        panel_height = (
            20
            +
            len(overlay_lines) * 24
        )

        # Black background panel
        cv2.rectangle(
            frame,
            (8, 8),
            (475, panel_height),
            (0, 0, 0),
            -1
        )

        # Write information
        for (
            line_number,
            line
        ) in enumerate(
            overlay_lines
        ):

            cv2.putText(
                frame,
                line,
                (
                    18,
                    34 +
                    line_number * 24
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        # ====================================================
        # LIVE COLAB DISPLAY
        # ====================================================

        if (
            frame_number
            %
            DISPLAY_EVERY_N_FRAMES
            == 0
        ):

            clear_output(
                wait=True
            )

            cv2_imshow(frame)

            # Progress information
            if total_frames > 0:

                progress = (
                    frame_number
                    /
                    total_frames
                ) * 100

                print(
                    f"Processing: "
                    f"{progress:.1f}%"
                )

            print(
                f"Frame: "
                f"{frame_number}/"
                f"{total_frames}"
            )

            print(
                f"Vehicles detected: "
                f"{current_vehicle_count}"
            )

            print(
                f"Traffic status: "
                f"{traffic_status}"
            )

    # ========================================================
    # CLEANUP
    # ========================================================

    capture.release()

    elapsed_time = (
        time.time()
        -
        start_time
    )

    clear_output(
        wait=True
    )

    print("==========================================")
    print("✅ TRAFFIC ANALYSIS COMPLETE")
    print("==========================================")
    print(
        f"Frames processed: {frame_number}"
    )
    print(
        f"Processing time: {elapsed_time:.1f} seconds"
    )
    print("No output video was saved.")
    print("==========================================")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()