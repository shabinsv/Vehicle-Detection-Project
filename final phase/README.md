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

## Web upload dashboard (React + Node.js)

The local dashboard lets you upload a video and see YOLO's annotated frames and traffic metrics in your browser while the video is processed. It runs locally; video files are saved temporarily in `uploads` and removed when processing ends. Only one video can be processed at a time.

Install Node.js, then in this folder install the web dependencies:

```bash
npm install
```

In one terminal, start the Node.js server:

```bash
npm run server
```

In a second terminal, start the React development server:

```bash
npm run dev
```

Open <http://localhost:5173>, choose a video, and click **Start analysis**. Click **Stop analysis** to end processing early. The Python packages in `requirements.txt` must be installed first, and the pretrained model file must be available (it downloads automatically on first use if internet access is available). If your Python command isn't `python` on Windows or `python3` elsewhere, set `PYTHON_COMMAND` to the correct executable before starting the Node server.

The web view is for uploaded video files, not a live camera feed. Processing happens frame by frame in Python; the annotated frames and actual metrics are streamed to the browser as they are produced.

### Windows install error: `WinError 206`

If `pip install -r requirements.txt` fails while installing PyTorch with `WinError 206` (path too long), the other packages may already have installed. Open PowerShell in this folder and use a short temporary path, then install only the missing YOLO package:

```powershell
New-Item -ItemType Directory -Force C:\tmp | Out-Null
$env:TEMP = 'C:\tmp'
$env:TMP = 'C:\tmp'
python -m pip install ultralytics --no-deps
```

Then check the imports:

```powershell
python -c "import cv2, numpy, torch, ultralytics; print('Python dependencies are ready')"
```

If that reports another missing package, install it with `python -m pip install PACKAGE_NAME`. The packages to import are `cv2` (from `opencv-python`), `numpy`, `torch`, and `ultralytics`. Start `npm run server` from this same terminal so it uses the same Python found by the `python` command.

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
