"""
api.py (v3)
-----------
Run:  uvicorn api:app --host 0.0.0.0 --port 8000
Open: http://localhost:8000
"""
from __future__ import annotations
import base64, time
from pathlib import Path
from typing import Annotated

import cv2, numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from predictor import AgeGenderPredictor, AGE_BUCKETS

app = FastAPI(title="VisageAI API", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

predictor: AgeGenderPredictor | None = None

@app.on_event("startup")
async def load():
    global predictor
    predictor = AgeGenderPredictor()

class FaceResult(BaseModel):
    face_index:         int
    bounding_box:       dict[str, int]
    face_confidence:    float
    gender:             str
    gender_confidence:  float
    age_value:          int           # precise age (DEX weighted mean)
    age_range:          str           # e.g. "25–32"
    age_confidence:     float
    age_probabilities:  dict[str, float]

class PredictResponse(BaseModel):
    filename:       str
    faces_detected: int
    latency_ms:     float
    results:        list[FaceResult]
    annotated_b64:  str

@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": predictor is not None}

@app.get("/", response_class=HTMLResponse)
async def ui():
    p = Path(__file__).parent / "interface.html"
    if not p.exists():
        raise HTTPException(404, "interface.html not found")
    return HTMLResponse(p.read_text(encoding="utf-8"))

@app.post("/predict", response_model=PredictResponse)
async def predict(file: Annotated[UploadFile, File()]):
    if not predictor:
        raise HTTPException(503, "Models not loaded")
    contents = await file.read()
    frame = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Cannot decode image")

    t0      = time.perf_counter()
    results = predictor.predict(frame)
    elapsed = (time.perf_counter() - t0) * 1000

    annotated = predictor.draw(frame, results, show_bar=True)
    _, buf = cv2.imencode(".png", annotated)
    b64 = base64.b64encode(buf).decode()

    out = []
    for i, r in enumerate(results):
        x1,y1,x2,y2 = r["box"]
        out.append(FaceResult(
            face_index        = i+1,
            bounding_box      = {"x1":x1,"y1":y1,"x2":x2,"y2":y2},
            face_confidence   = round(r["face_conf"],4),
            gender            = r["gender"],
            gender_confidence = round(r["gender_conf"],4),
            age_value         = r["age_value"],
            age_range         = r["age_range"],
            age_confidence    = round(r["age_conf"],4),
            age_probabilities = {b:round(p,4) for b,p in zip(AGE_BUCKETS,r["age_probs"])},
        ))
    return PredictResponse(filename=file.filename or "upload",
                           faces_detected=len(results),
                           latency_ms=round(elapsed,2),
                           results=out, annotated_b64=b64)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)