# VisageAI — Gender Prediction & Age Estimation

> Real-time and batch-mode prediction of **gender** and **precise age** from facial images, powered entirely by OpenCV's DNN module — no PyTorch or TensorFlow required. Includes a full browser UI, REST API, webcam support, and batch CSV export.

---

## What's New in v3

| Area | Change |
|------|--------|
| **Age accuracy** | Switched from `argmax` bucket selection to **DEX-style weighted expectation** — outputs a precise integer age (e.g. `34`) instead of a vague range like `(25-32)` |
| **Browser UI** | Full consumer-grade web app at `http://localhost:8000` — drag-drop upload, animated age counter, canvas bounding boxes, Original/Annotated toggle, tips section, FAQ accordion |
| **API response** | Added `age_value` (int) and `age_range` (str) fields alongside existing bucket probabilities |
| **Face padding** | Increased from 0.20 → 0.25 for better crop accuracy |

---

## Models Used

| Task | Architecture | Dataset | Input |
|------|-------------|---------|-------|
| Face detection | ResNet-10 SSD | VGGFace2 | 300×300 |
| Gender classifier | GoogLeNet (Caffe) | Adience | 227×227 |
| Age estimator | GoogLeNet (Caffe) | Adience | 227×227 |

**Age buckets:** `(0-2)` `(4-6)` `(8-12)` `(15-20)` `(25-32)` `(38-43)` `(48-53)` `(60-100)`

**Age estimation method (DEX-style):**
```
age_value = Σ ( probability[i] × midpoint[i] )
midpoints = [1, 5, 10, 17, 28, 40, 50, 70]
```

---

## Project Structure

```
age_gender_project/
├── models/                           # Downloaded model weights (not in git)
│   ├── deploy.prototxt
│   ├── res10_300x300_ssd_iter_140000.caffemodel
│   ├── age_deploy.prototxt
│   ├── age_net.caffemodel
│   ├── gender_deploy.prototxt
│   └── gender_net.caffemodel
│
├── predictor.py          # Core ML engine — face detect + age + gender
├── api.py                # FastAPI server — serves browser UI + REST endpoints
├── interface.html        # Browser UI (served at http://localhost:8000) 
├── download_models.py    # One-time model weight downloader
└── requirements.txt
```

**Minimum files to run the browser app:**
```
predictor.py  +  api.py  +  interface.html  +  models/
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download model weights (~30 MB)
```bash
python download_models.py
```

### 3. Launch the browser app
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```
Open **http://localhost:8000**

### 4. Or run on a single image (CLI)
```bash
python predict_image.py photo.jpg
```

### 5. Or run live webcam
```bash
python predict_webcam.py
# Q = quit   |   S = save screenshot
```

---

## Usage Examples

### Browser UI features
- Drag-drop or click-to-browse upload
- Sensitivity slider (face detection confidence threshold)
- Animated age counter (counts up to result)
- Age category label (Young Adult, Senior, etc.)
- Gender with confidence badge
- Animated confidence bars
- Age probability distribution chart across all 8 buckets
- Original / Annotated image toggle
- Download annotated PNG
- Tips section and FAQ accordion

### Image prediction (CLI)
```bash
# Single image
python predict_image.py photo.jpg

# Stricter detection, hide probability bar
python predict_image.py group.jpg --conf 0.7 --no-bar

# Process entire folder
python predict_image.py photos/
```

### Webcam (CLI)
```bash
python predict_webcam.py
python predict_webcam.py --camera 1    # secondary camera
python predict_webcam.py --no-bar      # cleaner overlay
```

### Batch evaluation + CSV
```bash
python batch_evaluate.py --input photos/ --output results.csv --save-annotated
```

Output CSV columns:
```
image, face_index, face_conf, gender, gender_conf, age_bucket, age_conf,
latency_ms, age_(0-2), age_(4-6), age_(8-12), age_(15-20),
age_(25-32), age_(38-43), age_(48-53), age_(60-100)
```

### REST API
```bash
# Start server
uvicorn api:app --host 0.0.0.0 --port 8000

# Swagger UI
open http://localhost:8000/docs

# Health check
curl http://localhost:8000/health

# Predict
curl -X POST http://localhost:8000/predict \
     -F "file=@photo.jpg"
```

**JSON response (v3):**
```json
{
  "filename": "photo.jpg",
  "faces_detected": 1,
  "latency_ms": 38.4,
  "annotated_b64": "<base64 PNG>",
  "results": [
    {
      "face_index": 1,
      "bounding_box": {"x1": 120, "y1": 80, "x2": 280, "y2": 300},
      "face_confidence": 0.9981,
      "gender": "Female",
      "gender_confidence": 0.9723,
      "age_value": 31,
      "age_range": "25–32",
      "age_confidence": 0.8841,
      "age_probabilities": {
        "(0-2)": 0.0001,
        "(4-6)": 0.0003,
        "(8-12)": 0.0012,
        "(15-20)": 0.0088,
        "(25-32)": 0.8841,
        "(38-43)": 0.0921,
        "(48-53)": 0.0112,
        "(60-100)": 0.0022
      }
    }
  ]
}
```

---

## Using the predictor in your own code

```python
import cv2
from predictor import AgeGenderPredictor

predictor = AgeGenderPredictor(face_confidence=0.5)

frame   = cv2.imread("photo.jpg")
results = predictor.predict(frame)

for r in results:
    print(r["gender"],    r["gender_conf"])   # "Female", 0.97
    print(r["age_value"], r["age_range"])     # 31, "25–32"
    print(r["age_conf"],  r["age_probs"])     # 0.88, [0.0001, ..., 0.8841, ...]

# Draw bounding boxes + labels + age bar chart onto frame
output = predictor.draw(frame, results)
cv2.imwrite("output.jpg", output)
```

---

## How it works

```
Input image / frame
      │
      ▼
┌──────────────────────────────────────┐
│  Face Detector  (ResNet-10 SSD)      │  → bounding boxes + confidence
│  Input : 300×300 blob                │
│  Mean  : (104, 177, 123)             │
└──────────────────────────────────────┘
      │  for each face crop (+25% padding)
      ▼
┌──────────────────────────────────────┐
│  Gender Classifier  (GoogLeNet)      │  → Male | Female + confidence
│  Input : 227×227 blob                │
│  Mean  : (78.4, 87.8, 114.9)        │
└──────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────┐
│  Age Estimator  (GoogLeNet)          │  → softmax over 8 buckets
│  Input : 227×227 blob                │  → DEX weighted mean → integer age
│  Mean  : (78.4, 87.8, 114.9)        │
└──────────────────────────────────────┘
      │
      ▼
  result dict  +  annotated frame  +  base64 PNG (API only)
```

---

## Key Design Decisions

**DEX-style age estimation** — instead of `argmax` (picks the single highest bucket,
ignores all other signal), we compute the expected value:
```python
age_value = int(round(np.dot(softmax_probs, [1, 5, 10, 17, 28, 40, 50, 70])))
```
If the model is 45% confident in `(25-32)` and 40% in `(38-43)`, argmax returns 28
while DEX returns ~34 — a much more realistic estimate.

**Face padding = 0.25** — the Adience models were trained with visible forehead, chin,
and cheeks in frame. A tight crop gives poor results; 25% padding restores the
expected context.

**Mean subtraction** — the face detector uses `(104, 177, 123)` while the age/gender
GoogLeNet uses `(78.4, 87.8, 114.9)`. Mixing them up is the single most common cause
of near-random predictions.

**Single model load at startup** — models load once via `@app.on_event("startup")`.
Loading ~30 MB of weights on every request would add 2–5 seconds per call.

---

## Confidence Threshold Guide

| Value | Behaviour | Best for |
|-------|-----------|----------|
| `0.3` | More detections, higher false-positive rate | Group photos, low quality |
| `0.5` | Balanced — recommended default | General use |
| `0.7+` | Strict — only high-confidence detections | Single face, authentication |

---

## Limitations

- Age outputs a **weighted estimate** — mean absolute error is roughly ±4–6 years on the Adience test set.
- Gender prediction is binary (Male/Female) — reflects the Adience training labels.
- Accuracy drops with face angles >45°, heavy occlusion, very low resolution (<60×60 px crop), or strong lighting contrast.
- Models run on CPU by default. For GPU acceleration, rebuild OpenCV with CUDA and set:
  ```python
  net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
  net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
  ```

---

## License

Model weights are from OpenCV's model zoo and the Adience benchmark — see their respective licenses.
Project code is MIT.
