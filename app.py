import re
import shutil
import subprocess
from pathlib import Path

import librosa
import numpy as np
import streamlit as st
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector

st.set_page_config(page_title="AI Video Clipper Pro", layout="wide")

# Safety check for ffmpeg
if shutil.which("ffmpeg") is None:
    st.error("ffmpeg not found on server. Try again or contact dev.")
    st.stop()

BASE_DIR = Path(__file__).parent
DOWNLOADS_DIR = BASE_DIR / "downloads"
CLIPS_DIR = BASE_DIR / "clips"
TEMP_DIR = BASE_DIR / "temp"

for folder in [DOWNLOADS_DIR, CLIPS_DIR, TEMP_DIR]:
    folder.mkdir(exist_ok=True)

st.title("AI Video Clipper Pro")
st.write("Upload a video (mp4 or mov) and generate smart clips. Max ~200MB recommended on free tier.")

uploaded_file = st.file_uploader(
    "Choose video file",
    type=["mp4", "mov"],
    help="Short videos work best (under 5 min) on free tier."
)

max_clips = st.slider("How many clips?", 1, 10, 5)
vertical_mode = st.checkbox("Make 9:16 Shorts", value=True)

def safe_name(text: str) -> str:
    text = re.sub(r'[\\/*?:"<>|]', "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:80] or "video"

def clear_folder(folder: Path):
    for item in folder.iterdir():
        try:
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
        except Exception:
            pass

def run_cmd(cmd):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Command failed: {e.stderr.strip() or str(e)}")

def get_video_duration(video_path: str) -> float:
    return float(run_cmd([
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]))

def detect_scenes(video_path: str):
    try:
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=27.0))
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()

        scenes = []
        for scene in scene_list:
            start_sec = scene[0].get_seconds()
            end_sec = scene[1].get_seconds()
            scenes.append((start_sec, end_sec))
        return scenes
    except Exception:
        return []

def extract_audio_wav(video_path: str, wav_path: str):
    run_cmd([
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        wav_path
    ])

def get_audio_peaks(wav_path: str, hop_length: int = 512):
    try:
        y, sr = librosa.load(wav_path, sr=16000, mono=True)
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

        mean_rms = float(np.mean(rms))
        std_rms = float(np.std(rms))
        threshold = mean_rms + std_rms * 1.25

        peaks = []
        for t, val in zip(times, rms):
            if val > threshold:
                peaks.append((float(t), float(val)))

        return peaks
    except Exception:
        return []

def audio_score_for_range(start: float, end: float, peaks):
    vals = [peak_val for peak_time, peak_val in peaks if start <= peak_time <= end]
    if not vals:
        return 0
    return min(5, int(round(sum(vals) / len(vals) * 50)))

def build_candidates_from_scenes(scenes, peaks, video_duration):
    candidates = []

    if not scenes:
        step = 20
        t = 0
        while t < min(video_duration, step * 12):
            candidates.append({
                "start": float(t),
                "end": float(min(video_duration, t + 20)),
                "score": 1.0,
                "text": "Auto clip"
            })
            t += step
        return candidates

    for s_start, s_end in scenes:
        clip_start = max(0, s_start - 1)
        clip_end = min(video_duration, s_end + 2)

        if clip_end - clip_start < 8:
            clip_end = min(video_duration, clip_start + 12)
        if clip_end - clip_start > 35:
            clip_end = clip_start + 35

        aud_score = audio_score_for_range(clip_start, clip_end, peaks)
        scene_len_score = 2 if 8 <= (clip_end - clip_start) <= 30 else 1
        final_score = aud_score * 0.6 + scene_len_score * 0.4

        candidates.append({
            "start": round(clip_start, 2),
            "end": round(clip_end, 2),
            "score": round(final_score, 2),
            "text": "Scene-based clip"
        })

    return candidates

def overlaps(a_start, a_end, b_start, b_end):
    return not (a_end <= b_start or a_start >= b_end)

def pick_best_candidates(candidates, max_count=5):
    ranked = sorted(candidates, key=lambda x: x["score"], reverse=True)
    chosen = []

    for c in ranked:
        if any(overlaps(c["start"], c["end"], x["start"], x["end"]) for x in chosen):
            continue
        chosen.append(c)
        if len(chosen) >= max_count:
            break

    return sorted(chosen, key=lambda x: x["start"])

def cut_clip(video_path: str, start: float, end: float, output_path: str, vertical=False):
    vf_parts = []

    if vertical:
        vf_parts.append("scale=1080:1920:force_original_aspect_ratio=decrease")
        vf_parts.append("pad=1080:1920:(ow-iw)/2:(oh-ih)/2")

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start),
        "-to", str(end),
        "-i", video_path,
    ]

    if vf_parts:
        cmd += ["-vf", ",".join(vf_parts)]

    cmd += [
        "-c:v", "libx264",
        "-c:a", "aac",
        output_path
    ]

    run_cmd(cmd)

def process_uploaded_video(uploaded_file):
    if uploaded_file is None:
        return None, None

    clear_folder(DOWNLOADS_DIR)

    clean_name = safe_name(uploaded_file.name)
    video_path = str(DOWNLOADS_DIR / clean_name)

    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    title = safe_name(uploaded_file.name.rsplit(".", 1)[0])
    return video_path, title

if st.button("Generate Clips"):
    if uploaded_file is None:
        st.error("Upload a video file first.")
        st.stop()

    if uploaded_file.size > 200 * 1024 * 1024:
        st.error("File too large (>200MB). Free tier may crash or timeout.")
        st.stop()

    status = st.empty()
    try:
        clear_folder(CLIPS_DIR)
        clear_folder(TEMP_DIR)

        status.info("Processing uploaded video...")
        video_path, title = process_uploaded_video(uploaded_file)

        status.info("Detecting scenes...")
        scenes = detect_scenes(video_path)

        status.info("Analyzing audio spikes...")
        wav_path = str(TEMP_DIR / "audio.wav")
        extract_audio_wav(video_path, wav_path)
        peaks = get_audio_peaks(wav_path)

        status.info("Scoring best moments...")
        video_duration = get_video_duration(video_path)
        candidates = build_candidates_from_scenes(scenes, peaks, video_duration)
        best = pick_best_candidates(candidates, max_count=max_clips)

        if not best:
            st.error("No clips found.")
            st.stop()

        report_rows = []
        clip_paths = []

        status.info("Creating clips...")
        for i, clip in enumerate(best, start=1):
            out_path = CLIPS_DIR / f"{i:02d}_{safe_name(title)}.mp4"

            cut_clip(
                video_path=video_path,
                start=clip["start"],
                end=clip["end"],
                output_path=str(out_path),
                vertical=vertical_mode
            )

            report_rows.append({
                "Clip": i,
                "Start": clip["start"],
                "End": clip["end"],
                "Score": clip["score"],
            })
            clip_paths.append(out_path)

        status.success("Done. Clips created.")

        st.subheader("Generated Clips")
        for row, path in zip(report_rows, clip_paths):
            st.markdown(f"**Clip {row['Clip']}**")
            st.caption(f"{row['Start']}s → {row['End']}s | Score: {row['Score']}")
            st.video(str(path))

        st.subheader("Clip Report")
        st.json(report_rows)

    except Exception as e:
        status.empty()
        st.error(f"Error: {e}")
