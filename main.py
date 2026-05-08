from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, resample
import tempfile
import os

app = FastAPI(title="HRV Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Signal Processing ----------

def bandpass_filter(signal, fs, low=0.7, high=3.5, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def check_finger_contact(raw_signal: list) -> tuple[bool, str]:
    """
    Detect if a finger is actually covering the camera lens.
    Relaxed thresholds to accommodate different phone cameras and skin tones.
    """
    signal = np.array(raw_signal)

    mean_val = float(np.mean(signal))

    # RELAXED: was 150 — many valid readings (darker skin tones, older phones) fall below this
    if mean_val < 80:
        return False, (
            f"No finger detected (brightness={mean_val:.0f}). "
            "Press your fingertip firmly over the camera lens and flashlight."
        )

    # Saturation check — unchanged, this is a hard ceiling
    if mean_val > 252:
        return False, (
            "Camera appears fully saturated. "
            "Cover the lens completely with your fingertip — don't just hold it nearby."
        )

    # RELAXED: was 0.3 — low-variance signals can still carry a valid pulse
    ac_component = float(np.std(signal))
    if ac_component < 0.1:
        return False, (
            "Signal too flat — no pulse detected. "
            "Press your fingertip more firmly over the camera lens."
        )

    return True, "ok"


def check_signal_stationarity(raw_signal: list) -> tuple[bool, str]:
    """Check that the signal doesn't have huge jumps (finger lifted mid-recording)."""
    signal = np.array(raw_signal)
    chunks = np.array_split(signal, 4)
    means = [np.mean(c) for c in chunks]

    # RELAXED: was 40 — allows for mild finger shifts without rejecting the whole reading
    if max(means) - min(means) > 70:
        return False, (
            "Signal unstable — finger was lifted or moved during recording. "
            "Keep your fingertip still and pressed firmly over the lens for the full duration."
        )
    return True, "ok"


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

    raw_red = []
    raw_green = []
    raw_blue = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape
        roi = frame[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
        raw_blue.append(float(np.mean(roi[:, :, 0])))
        raw_green.append(float(np.mean(roi[:, :, 1])))
        raw_red.append(float(np.mean(roi[:, :, 2])))
    cap.release()

    # RELAXED: was 5 seconds — allow slightly shorter recordings
    if len(raw_red) < target_fps * 4:
        raise ValueError("Video too short — needs at least 4 seconds of recording.")

    # --- Finger contact checks ---
    ok, msg = check_finger_contact(raw_red)
    if not ok:
        raise ValueError(msg)

    ok, msg = check_signal_stationarity(raw_red)
    if not ok:
        raise ValueError(msg)

    # --- Channel dominance check ---
    mean_red = np.mean(raw_red)
    mean_green = np.mean(raw_green)
    mean_blue = np.mean(raw_blue)

    # RELAXED: was mean_green + 20 — some cameras and skin tones have tighter channel separation
    if mean_red < mean_green:
        raise ValueError(
            "No finger detected — red channel not dominant. "
            "Place your fingertip directly over the camera lens and flashlight."
        )

    # --- Signal processing ---
    duration = total_frames / fps
    target_frames = int(duration * target_fps)
    resampled_signal = resample(raw_red, target_frames)
    resampled_time = np.linspace(0, duration, target_frames)

    filtered_signal = bandpass_filter(resampled_signal, target_fps)

    # Adaptive peak detection based on signal amplitude
    signal_range = np.max(filtered_signal) - np.min(filtered_signal)

    # RELAXED: was 0.15 — lower prominence threshold catches subtler pulse peaks
    min_prominence = signal_range * 0.08

    peaks, props = find_peaks(
        filtered_signal,
        distance=int(target_fps * 0.35),   # max ~170 BPM
        prominence=min_prominence,
    )

    rr_intervals = np.diff(resampled_time[peaks])
    # Only keep physiologically valid RR intervals (30-200 BPM)
    rr_intervals = rr_intervals[(rr_intervals > 0.3) & (rr_intervals < 2.0)]

    # RELAXED: was 0.35 — healthy HRV and real-world noise can exceed 0.35 legitimately
    if len(rr_intervals) > 2:
        rr_cv = float(np.std(rr_intervals) / np.mean(rr_intervals))
        if rr_cv > 0.55:
            raise ValueError(
                "Irregular signal detected — this doesn't look like a heartbeat. "
                "Press your fingertip firmly and stay still."
            )

    expected_peaks = int(duration / 0.75)
    signal_quality = min(100.0, (len(peaks) / max(expected_peaks, 1)) * 100)

    # Waveform for display
    waveform_points = 200
    if len(filtered_signal) > waveform_points:
        waveform = resample(filtered_signal, waveform_points).tolist()
    else:
        waveform = filtered_signal.tolist()

    # RELAXED: rr_intervals threshold was 5, quality threshold was 60%
    if len(rr_intervals) >= 3 and signal_quality >= 40:
        hrv = compute_hrv(rr_intervals)
        hrv["signal_quality_pct"] = round(signal_quality, 1)
        hrv["waveform"] = waveform
        return hrv
    else:
        raise ValueError(
            f"Could not detect enough heartbeats (quality={signal_quality:.0f}%). "
            "Ensure your fingertip covers the camera lens fully, the flash is on, and stay still."
        )


# ---------- Routes ----------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(video: UploadFile = File(...)):
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
