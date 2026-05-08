from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, resample
import tempfile
import os
import json

app = FastAPI(title="HRV Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Signal Processing ----------

def bandpass_filter(signal, fs, low=0.7, high=3.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def compute_hrv(rr_intervals):
    rr_ms = np.array(rr_intervals) * 1000
    diffs = np.diff(rr_ms)
    sdnn = float(np.std(rr_ms))
    rmssd = float(np.sqrt(np.mean(diffs ** 2))) if len(diffs) > 0 else 0.0
    pnn50 = float(np.sum(np.abs(diffs) > 50) / len(diffs) * 100) if len(diffs) > 0 else 0.0
    return {
        "mean_rr_ms": float(np.mean(rr_ms)),
        "mean_hr_bpm": float(60000.0 / np.mean(rr_ms)),
        "sdnn_ms": sdnn,
        "rmssd_ms": rmssd,
        "pnn50_pct": pnn50,
        "peaks_detected": len(rr_intervals) + 1,
    }


def process_video_file(video_path: str, target_fps: int = 30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    raw_signal = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape
        roi = frame[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
        red_avg = float(np.mean(roi[:, :, 2]))
        raw_signal.append(red_avg)
    cap.release()

    if len(raw_signal) < target_fps * 3:
        raise ValueError("Video too short — needs at least 3 seconds")

    duration = total_frames / fps
    target_frames = int(duration * target_fps)
    resampled_signal = resample(raw_signal, target_frames)
    resampled_time = np.linspace(0, duration, target_frames)

    filtered_signal = bandpass_filter(resampled_signal, target_fps)

    peaks, _ = find_peaks(filtered_signal, distance=target_fps * 0.4)
    rr_intervals = np.diff(resampled_time[peaks])
    rr_intervals = rr_intervals[(rr_intervals > 0.3) & (rr_intervals < 2.0)]

    expected_peaks = int(duration / 0.8)
    signal_quality = min(100.0, (len(peaks) / max(expected_peaks, 1)) * 100)

    # Downsample waveform to ~200 points for the app
    waveform_points = 200
    if len(filtered_signal) > waveform_points:
        waveform = resample(filtered_signal, waveform_points).tolist()
    else:
        waveform = filtered_signal.tolist()

    if len(rr_intervals) > 3 and signal_quality >= 50:
        hrv = compute_hrv(rr_intervals)
        hrv["signal_quality_pct"] = round(signal_quality, 1)
        hrv["waveform"] = waveform
        return hrv
    else:
        raise ValueError(
            f"Low signal quality ({signal_quality:.0f}%). "
            "Ensure fingertip covers the camera lens fully and there is good lighting."
        )


# ---------- Routes ----------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(video: UploadFile = File(...)):
    # Validate file type
    if not video.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
        raise HTTPException(400, "Unsupported file type. Send MP4, MOV, AVI, or MKV.")

    suffix = os.path.splitext(video.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await video.read())
        tmp_path = tmp.name

    try:
        result = process_video_file(tmp_path)
        return result
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(500, f"Processing error: {str(e)}")
    finally:
        os.unlink(tmp_path)
