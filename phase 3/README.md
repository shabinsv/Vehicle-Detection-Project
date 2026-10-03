# AI-Based Road Traffic Congestion Detection Using YOLO

A small local Python project that detects and tracks cars, buses, trucks, and motorcycles in a video. It counts unique tracking IDs, estimates speed from tracked movement, measures approximate frame occupancy, and applies simple configurable rules to label traffic.

## Installation

Install Python, open a terminal in this folder, and run:

```bash
pip install -r requirements.txt
```

The pretrained Ultralytics model is downloaded automatically the first time the program runs. Internet access is needed for that first download.

## Video path

Open `traffic_congestion.py` and change:

```python
VIDEO_PATH = "traffic.mp4"
```

to the path of your video, for example:

```python
VIDEO_PATH = r"D:\traffic_project\traffic.mp4"
```

## Run

```bash
python traffic_congestion.py
```

The processed video appears in an OpenCV window. Press `q` to close it.

## How it works

```text
Video
  ↓
YOLO
  ↓
Vehicle Detection
  ↓
Vehicle Tracking
  ↓
Vehicle Counting
  ↓
Speed Estimation
  ↓
Occupancy
  ↓
Congestion Classification
```

YOLO uses a pretrained Ultralytics model and ByteTrack to assign vehicle IDs. Counts are based on unique IDs seen so far. Speed is an approximate estimate from a tracked vehicle's center movement, video frame rate, and the configurable `PIXEL_TO_METER` value; it is shown as `N/A` until enough track points and a valid frame rate are available. Bounding-box area gives only an approximate occupancy value. Congestion thresholds are simple Academy-project settings, not official traffic standards. Change these constants near the top of the Python file to adjust the calculations.
