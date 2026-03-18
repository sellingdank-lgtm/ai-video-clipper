import re
import shutil
import subprocess
from pathlib import Path
import librosa
import numpy as np
import streamlit as st
import yt_dlp
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector

st.set_page_config(page_title="AI Video Clipper Pro", layout="wide")

# === CHECK FFMPEG ===
if shutil.which("ffmpeg") is None:
    st.error("ffmpeg not found on server.")
    st.stop()

BASE_DIR = Path(__file__).parent
DOWNLOADS_DIR = BASE_DIR / "downloads"
CLIPS_DIR = BASE_DIR / "clips"
TEMP_DIR = BASE_DIR / "temp"

for folder in [DOWNLOADS_DIR, CLIPS_DIR, TEMP_DIR]:
    folder.mkdir(exist_ok=True)

st.title("AI Video Clipper Pro")
st.write("Paste a YouTube link and generate smart clips.")

url = st.text_input("YouTube Link")
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
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]))

def download_video(youtube_url: str):
    clear_folder(DOWNLOADS_DIR)

    ydl_opts = {
        "format": "bv*+ba/best",
        "merge_output_format": "mp4",
        "outtmpl": str(DOWNLOADS_DIR / "%(title)s.%(ext)s"),
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        title = safe_name(info.get("title", "video"))

    mp4s = sorted(DOWNLOADS_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not mp4s:
        raise FileNotFoundError("Video download failed.")
    return str(mp4s[0]), title

def detect_scenes(video_path: str):
    try:
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=27.0))
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()

        scenes = []
        for scene in scene_list:
            scenes.append((scene[0].get_seconds(), scene[1].get_seconds()))
        return scenes
    except:
        return []

def extract_audio_wav(video_path: str, wav_path: str):
    run_cmd([
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        wav_path
    ])

def get_audio_peaks(wav_path: str):
    try:
        y, sr = librosa.load(wav_path, sr=16000, mono=True)
        rms = librosa.feature.rms(y=y)[0]
        times = librosa.frames_to_time(range(len(rms)), sr=sr)

        threshold = np.mean(rms) + np.std(rms) * 1.25

        peaks = []
        for t, val in zip(times, rms):
            if val > threshold:
                peaks.append((float(t), float(val)))
        return peaks
    except:
        return []

def audio_score(start, end, peaks):
    vals = [v for t, v in peaks if start <= t <= end]
    return min(5, int(sum(vals) * 50)) if vals else 0

def build_candidates(scenes, peaks, duration):
    if not scenes:
        return [{"start": i, "end": i+20, "score": 1} for i in range(0, int(duration), 20)]

    candidates = []
    for s, e in scenes:
        start = max(0, s-1)
        end = min(duration, e+2)

        score = audio_score(start, end, peaks)
        candidates.append({"start": start, "end": end, "score": score})
    return candidates

def pick_best(candidates, max_count):
    return sorted(candidates, key=lambda x: x["score"], reverse=True)[:max_count]

def cut_clip(video_path, start, end, output_path, vertical):
    vf = []
    if vertical:
        vf.append("scale=1080:1920:force_original_aspect_ratio=decrease")
        vf.append("pad=1080:1920:(ow-iw)/2:(oh-ih)/2")

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-to", str(end),
        "-i", video_path
    ]

    if vf:
        cmd += ["-vf", ",".join(vf)]

    cmd += ["-c:v", "libx264", "-c:a", "aac", output_path]

    run_cmd(cmd)

if st.button("Generate Clips"):
    try:
        clear_folder(CLIPS_DIR)
        clear_folder(TEMP_DIR)

        st.info("Downloading...")
        video_path, title = download_video(url)

        st.info("Scenes...")
        scenes = detect_scenes(video_path)

        st.info("Audio...")
        wav = str(TEMP_DIR / "audio.wav")
        extract_audio_wav(video_path, wav)
        peaks = get_audio_peaks(wav)

        duration = get_video_duration(video_path)
        candidates = build_candidates(scenes, peaks, duration)
        best = pick_best(candidates, max_clips)

        st.info("Cutting clips...")
        clips = []
        for i, c in enumerate(best, 1):
            out = CLIPS_DIR / f"{i}_{title}.mp4"
            cut_clip(video_path, c["start"], c["end"], str(out), vertical_mode)
            clips.append(out)

        st.success("Done")

        for c in clips:
            st.video(str(c))

    except Exception as e:
        st.error(str(e))