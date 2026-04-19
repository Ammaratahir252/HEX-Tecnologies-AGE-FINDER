
from __future__ import annotations
import cv2
import numpy as np

AGE_BUCKETS  = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
AGE_RANGES   = ["0–2",   "4–6",   "8–12",   "15–20",   "25–32",   "38–43",   "48–53",   "60+"]

# Midpoint of each bucket — used for weighted-mean age estimation (DEX-style)
AGE_MIDPOINTS = np.array([1, 5, 10, 17, 28, 40, 50, 70], dtype=np.float32)

GENDER_LABELS = ["Male", "Female"]
MODEL_MEAN    = (78.4263377603, 87.7689143744, 114.895847746)


class AgeGenderPredictor:
    def __init__(
        self,
        face_prototxt:   str   = "models/deploy.prototxt",
        face_model:      str   = "models/res10_300x300_ssd_iter_140000.caffemodel",
        age_prototxt:    str   = "models/age_deploy.prototxt",
        age_model:       str   = "models/age_net.caffemodel",
        gender_prototxt: str   = "models/gender_deploy.prototxt",
        gender_model:    str   = "models/gender_net.caffemodel",
        face_confidence: float = 0.5,
        face_padding:    float = 0.25,
    ) -> None:
        self.face_confidence = face_confidence
        self.face_padding    = face_padding

        print("Loading face detector ...", end=" ", flush=True)
        self.face_net = cv2.dnn.readNetFromCaffe(face_prototxt, face_model)
        print("✓")
        print("Loading age model ...", end=" ", flush=True)
        self.age_net = cv2.dnn.readNetFromCaffe(age_prototxt, age_model)
        print("✓")
        print("Loading gender model ...", end=" ", flush=True)
        self.gender_net = cv2.dnn.readNetFromCaffe(gender_prototxt, gender_model)
        print("✓")

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict(self, frame: np.ndarray) -> list[dict]:
        """
        Returns list of dicts:
          box, face_conf,
          gender, gender_conf,
          age_value (int), age_range (str), age_conf (float), age_probs (list)
        """
        faces   = self._detect_faces(frame)
        results = []
        for box, face_conf in faces:
            crop = self._crop_face(frame, box)
            gender, gender_conf           = self._predict_gender(crop)
            age_val, age_range, age_conf, age_probs = self._predict_age(crop)
            results.append({
                "box":         box,
                "face_conf":   face_conf,
                "gender":      gender,
                "gender_conf": gender_conf,
                "age_value":   age_val,       # e.g. 28  ← precise number
                "age_range":   age_range,     # e.g. "25–32"
                "age_conf":    age_conf,
                "age_probs":   age_probs,
            })
        return results

    def draw(self, frame: np.ndarray, results: list[dict], show_bar: bool = True) -> np.ndarray:
        out = frame.copy()
        for r in results:
            x1, y1, x2, y2 = r["box"]
            color = (86, 180, 233) if r["gender"] == "Male" else (230, 97, 0)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            label = f"{r['gender']} {r['gender_conf']:.0%}  |  Age ~{r['age_value']} ({r['age_range']})"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            tag_y = y1 - 12 if y1 > 30 else y2 + lh + 12
            cv2.rectangle(out, (x1, tag_y - lh - 8), (x1 + lw + 10, tag_y + 4), color, -1)
            cv2.putText(out, label, (x1 + 5, tag_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            if show_bar:
                out = self._draw_age_bar(out, r, x1, y2)
        return out

    # ── Internal ──────────────────────────────────────────────────────────────

    def _detect_faces(self, frame: np.ndarray) -> list[tuple]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                                     (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        dets = self.face_net.forward()
        faces = []
        for i in range(dets.shape[2]):
            conf = float(dets[0, 0, i, 2])
            if conf < self.face_confidence:
                continue
            box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            faces.append(((max(0,x1), max(0,y1), min(w,x2), min(h,y2)), conf))
        return faces

    def _crop_face(self, frame: np.ndarray, box: tuple) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        pw = int((x2 - x1) * self.face_padding)
        ph = int((y2 - y1) * self.face_padding)
        return frame[max(0,y1-ph):min(h,y2+ph), max(0,x1-pw):min(w,x2+pw)]

    def _blob(self, face: np.ndarray) -> np.ndarray:
        return cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN, swapRB=False)

    def _predict_gender(self, face: np.ndarray) -> tuple[str, float]:
        self.gender_net.setInput(self._blob(face))
        preds = self.gender_net.forward()[0]
        idx   = int(np.argmax(preds))
        return GENDER_LABELS[idx], float(preds[idx])

    def _predict_age(self, face: np.ndarray) -> tuple[int, str, float, list[float]]:
        self.age_net.setInput(self._blob(face))
        preds = self.age_net.forward()[0]               # shape (8,) softmax

        # ── DEX-style weighted mean ──────────────────────────────────────────
        # Instead of argmax (gives wrong bucket), take E[age] = Σ p_i * mid_i
        # This is far more accurate and gives a precise integer age.
        age_value = int(round(float(np.dot(preds, AGE_MIDPOINTS))))
        age_value = max(1, min(100, age_value))         # clamp 1–100

        # Best bucket for the range label
        idx       = int(np.argmax(preds))
        probs     = [float(p) for p in preds]
        return age_value, AGE_RANGES[idx], float(preds[idx]), probs

    @staticmethod
    def _draw_age_bar(frame, result, x, y_start):
        bar_w, bar_h, gap = 20, 60, 3
        total_w = len(AGE_BUCKETS) * (bar_w + gap) - gap
        ox, oy  = x, y_start + 8
        cv2.rectangle(frame, (ox-4, oy-4), (ox+total_w+4, oy+bar_h+20), (30,30,30), -1)
        for i, (label, prob) in enumerate(zip(AGE_BUCKETS, result["age_probs"])):
            bx     = ox + i * (bar_w + gap)
            filled = int(prob * bar_h)
            is_top = i == int(np.argmax(result["age_probs"]))
            cv2.rectangle(frame, (bx, oy), (bx+bar_w, oy+bar_h), (60,60,60), -1)
            color = (86,200,86) if is_top else (120,150,200)
            cv2.rectangle(frame, (bx, oy+bar_h-filled), (bx+bar_w, oy+bar_h), color, -1)
        for i, label in enumerate(AGE_BUCKETS):
            bx = ox + i*(bar_w+gap) + bar_w//2
            cv2.putText(frame, label[1:-1].split("-")[0], (bx-5, oy+bar_h+14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180,180,180), 1, cv2.LINE_AA)
        return frame