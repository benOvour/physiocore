import os, re, cv2, numpy as np, pickle, tempfile
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.requests import Request
from typing import List
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import urllib.request
import tensorflow as tf

app       = FastAPI(title="PhysioScore API")
templates = Jinja2Templates(directory="templates")

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "model", "physio_finetuned_100_v2.tflite")
CONFIG_PATH = os.path.join(BASE_DIR, "model", "config_final.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler_final.pkl")
MP_MODEL    = os.path.join(BASE_DIR, "model", "pose_landmarker.task")

# ── Download MediaPipe model if not present ───────────────────
if not os.path.exists(MP_MODEL):
    print("Downloading MediaPipe pose model...")
    os.makedirs(os.path.dirname(MP_MODEL), exist_ok=True)
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        MP_MODEL
    )
    print("MediaPipe model downloaded.")

# ── Load config + scaler + model ─────────────────────────────
print("Loading config and scaler...")
with open(CONFIG_PATH, "rb") as f: config = pickle.load(f)
with open(SCALER_PATH, "rb") as f: scaler = pickle.load(f)

ALL_OFFSETS     = config["MP_OFFSETS"]
MAX_LEN         = config["MAX_LEN"]
N_FEAT          = config["N_FEATURES"]
SELECTED_JOINTS = config["SELECTED_JOINTS"]
EXERCISE_NAMES  = config["EXERCISE_NAMES"]

print("Loading TFLite model...")
try:
    interpreter = tf.lite.Interpreter(
        model_path=MODEL_PATH,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.AUTO
    )
except Exception:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
IN_DET  = interpreter.get_input_details()
OUT_DET = interpreter.get_output_details()
print("Model loaded successfully!")

# ── Constants ─────────────────────────────────────────────────
EXERCISE_WEIGHTS = {
    0:1.5, 1:1.3, 2:1.2, 3:1.2,
    4:1.5, 5:1.3, 6:1.0, 7:1.0,
    8:1.1, 9:1.0,
}
FEEDBACK = {
    0: ("Maintain upright trunk, knees over feet",    "Good squat depth"),
    1: ("Keep hips and knees vertical on stance leg", "Good hip flexion"),
    2: ("Avoid lateral trunk deviation",              "Good lunge form"),
    3: ("Reduce knee valgus, keep trunk >30°",        "Good side lunge"),
    4: ("Avoid using arms, keep trunk upright",       "Good sit-to-stand"),
    5: ("Minimise pelvis deviation",                  "Good leg raise"),
    6: ("Raise arm higher, keep in frontal plane",    "Good shoulder abduction"),
    7: ("Extend arm further, keep in sagittal plane", "Good shoulder extension"),
    8: ("Increase rotation range both directions",    "Good shoulder rotation"),
    9: ("Raise arm higher, maintain scapular plane",  "Good shoulder scaption"),
}

# ── MediaPipe helpers ─────────────────────────────────────────
def extract_frame(landmarks):
    lm      = landmarks
    hip_x   = (lm[23].x + lm[24].x) / 2
    hip_y   = (lm[23].y + lm[24].y) / 2
    shldr_x = (lm[11].x + lm[12].x) / 2
    shldr_y = (lm[11].y + lm[12].y) / 2
    sl      = max(np.sqrt((shldr_x-hip_x)**2+(shldr_y-hip_y)**2), 1e-6)
    IDX     = {
        "Spine":11,"Chest":12,"LeftUpperArm":11,"LeftForearm":13,
        "RightUpperArm":12,"RightForearm":14,"LeftUpperLeg":23,
        "LeftLowerLeg":25,"RightUpperLeg":24,"RightLowerLeg":26,
    }
    pos = []
    for jn in SELECTED_JOINTS:
        i = IDX[jn]
        pos.extend([(lm[i].x-hip_x)/sl, (lm[i].y-hip_y)/sl])
    arr = np.array(pos, dtype=np.float32)
    for col, off in ALL_OFFSETS.items():
        arr[col] += off
    return arr

def process_video(path: str):
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MP_MODEL),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )
    cap, frames = cv2.VideoCapture(path), []
    with mp_vision.PoseLandmarker.create_from_options(opts) as det:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = det.detect(img)
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                frames.append(extract_frame(result.pose_landmarks[0]))
    cap.release()
    return np.array(frames, dtype=np.float32)

def predict(seq):
    sc = scaler.transform(seq)
    n  = len(sc)
    if n < MAX_LEN:
        sc = np.vstack([sc, np.zeros((MAX_LEN-n, N_FEAT), dtype=np.float32)])
    else:
        sc = sc[:MAX_LEN]
    interpreter.set_tensor(IN_DET[0]["index"], sc[np.newaxis].astype(np.float32))
    interpreter.invoke()
    o0    = interpreter.get_tensor(OUT_DET[0]["index"])[0]
    o1    = interpreter.get_tensor(OUT_DET[1]["index"])[0]
    cls_p = o0 if len(o0)==10 else o1
    scr   = o1[0] if len(o0)==10 else o0[0]
    return cls_p, float(scr)*100

# ── Routes ────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/assess", response_class=HTMLResponse)
async def assess_page(request: Request):
    return templates.TemplateResponse("assess.html", {"request": request})

@app.get("/results", response_class=HTMLResponse)
async def results_page(request: Request):
    return templates.TemplateResponse("results.html", {"request": request})

@app.post("/api/analyze")
async def analyze(videos: List[UploadFile] = File(...)):
    if len(videos) != 10:
        raise HTTPException(400, f"Need exactly 10 videos, got {len(videos)}")

    results = []
    for upload in videos:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(await upload.read())
            tmp_path = tmp.name
        try:
            seq = process_video(tmp_path)
            if len(seq) < 10:
                results.append({"error": "Too few frames", "filename": upload.filename})
                continue
            cls_p, quality = predict(seq)
            ex_idx         = int(np.argmax(cls_p))
            poor_fb, good_fb = FEEDBACK[ex_idx]
            status   = "good" if quality >= 70 else "fair" if quality >= 45 else "poor"
            feedback = good_fb if quality >= 70 else poor_fb
            results.append({
                "filename":      upload.filename,
                "exercise_idx":  ex_idx,
                "exercise_name": EXERCISE_NAMES[ex_idx+1],
                "confidence":    round(float(cls_p[ex_idx])*100, 1),
                "quality":       round(quality, 1),
                "status":        status,
                "feedback":      feedback,
                "all_probs":     {EXERCISE_NAMES[i+1]: round(float(p)*100,1)
                                  for i,p in enumerate(cls_p)},
            })
        finally:
            os.unlink(tmp_path)

    # Weighted recovery score
    total_w, total_s = 0, 0
    seen = set()
    for r in results:
        if "error" in r: continue
        idx = r["exercise_idx"]
        w   = EXERCISE_WEIGHTS.get(idx, 1.0) * (0.5 if idx in seen else 1.0)
        seen.add(idx)
        total_s += r["quality"] * w
        total_w += w

    recovery = round(total_s/total_w, 1) if total_w > 0 else 0

    if recovery >= 80:   level, msg = "Excellent",  "Outstanding movement quality across all exercises."
    elif recovery >= 65: level, msg = "Good",        "Most movements well controlled with minor areas to improve."
    elif recovery >= 50: level, msg = "Moderate",    "Several movements need attention. Continue rehabilitation."
    elif recovery >= 35: level, msg = "Fair",        "Significant movement limitations present. Focus on weak areas."
    else:                level, msg = "Needs Work",  "Multiple exercises require focused rehabilitation effort."

    return JSONResponse({
        "recovery_score": recovery,
        "level":          level,
        "message":        msg,
        "exercises":      results,
        "total_videos":   len(videos),
        "valid_videos":   len([r for r in results if "error" not in r]),
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)