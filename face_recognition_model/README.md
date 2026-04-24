# PC Face Recognition — Laptop Webcam

> **Branch:** `pc-compatible`
> This version removes the Intel NCS / Movidius hardware dependency and runs
> entirely on a standard Windows/macOS/Linux laptop using the `face_recognition`
> library (dlib-based).

---

## Quick Start

### 1. Install dependencies

```bash
pip install face_recognition opencv-python numpy
```

> **Windows note:** `face_recognition` requires `dlib`. The easiest way to get
> a pre-compiled dlib wheel on Windows is:
> ```bash
> pip install cmake
> pip install dlib
> pip install face_recognition
> ```
> Or use the conda-forge channel: `conda install -c conda-forge dlib face_recognition`.

### 2. Add your reference face(s)

Place one or more clear face photos inside the `validated_images/` folder.
Name each file after the person — the filename becomes the display label.

```
validated_images/
    alice.jpg       ->  shows "alice" when matched
    bob.png         ->  shows "bob"  when matched
    valid.jpg       ->  already present, rename to your name
```

Each image should contain **exactly one face** and be well-lit.

### 3. Run

```bash
cd face_recognition_model
python video_face_matcher.py
```

A webcam window opens. Known faces get a **green border + label**; unknown
faces get a **red border**.

---

## Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `S` | Save snapshot of current frame |
| `R` | Hot-reload reference faces from `validated_images/` |

---

## Configuration (top of `video_face_matcher.py`)

| Variable | Default | Description |
|----------|---------|-------------|
| `CAMERA_INDEX` | `0` | Webcam index (change if multiple cameras) |
| `FACE_MATCH_THRESHOLD` | `0.5` | Distance threshold — lower = stricter |
| `CONFIRMATION_FRAMES` | `2` | Consecutive frames required to confirm a match |
| `DETECTION_SCALE` | `0.5` | Resize factor before detection (higher = slower but more accurate) |

---

## How it works

1. Each frame from the webcam is scaled down for fast HOG-based face detection.
2. 128-dimension face embeddings are computed for every detected face.
3. Euclidean distance is compared against the pre-computed reference embeddings.
4. A match is confirmed only after `CONFIRMATION_FRAMES` consecutive positive frames.

---

## Migrating back to Raspberry Pi

When you are ready to deploy on the Pi, switch to the `main` branch which
contains the original Intel NCS / Movidius pipeline.
