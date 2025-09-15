import streamlit as st
import whisper
import os
from moviepy.editor import VideoFileClip, concatenate_videoclips
import torch
import tempfile
import numpy as np

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="AI Video Summarizer", layout="wide")
st.title("🎬 AI Video Summarizer & Short Video Generator")

# Upload video
video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

if video_file:
    # Save uploaded video
    video_path = f"uploaded_{video_file.name}"
    with open(video_path, "wb") as f:
        f.write(video_file.read())

    st.video(video_path)
    st.success("✅ Video uploaded successfully!")

    # ----------------------------
    # Transcribe Audio
    # ----------------------------
    st.info("⏳ Transcribing audio using Whisper...")
    model = whisper.load_model("base")  # tiny, base, small, medium, large
    result = model.transcribe(video_path)
    transcript = result["text"]

    st.subheader("📝 Full Transcription")
    st.text_area("Transcript", transcript, height=300)

    # ----------------------------
    # Generate AI-based short video summary
    # ----------------------------
    st.info("⏳ Creating AI short video summary...")

    clip = VideoFileClip(video_path)
    duration = int(clip.duration)

    # 1. Simple heuristic: select scenes where audio is above threshold
    def get_audio_energy(video_path, sample_rate=16000, chunk_duration=5):
        import subprocess
        import math

        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        # Extract audio as WAV
        cmd = f'ffmpeg -i "{video_path}" -ar {sample_rate} -ac 1 -y "{temp_audio}"'
        subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        import soundfile as sf
        audio, sr = sf.read(temp_audio)
        chunk_size = int(chunk_duration * sr)
        energies = []
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            energy = np.sum(chunk ** 2) / len(chunk)
            energies.append(energy)
        os.remove(temp_audio)
        return energies

    energies = get_audio_energy(video_path)
    # Take top 30% high-energy chunks as "important scenes"
    threshold = np.percentile(energies, 70)
    important_chunks = [i for i, e in enumerate(energies) if e >= threshold]

    timestamps = []
    for chunk_idx in important_chunks:
        start = chunk_idx * 5
        end = min((chunk_idx + 1) * 5, duration)
        timestamps.append((start, end))

    # Create short video
    if timestamps:
        clips = [clip.subclip(start, end) for start, end in timestamps]
        short_clip = concatenate_videoclips(clips)
        summary_path = f"short_{video_file.name}"
        short_clip.write_videofile(summary_path, codec="libx264", audio_codec="aac")

        st.subheader("🎞 AI Short Video Summary")
        st.video(summary_path)
        st.success("✅ Short video summary generated successfully!")
    else:
        st.warning("⚠️ Could not generate a summary: audio too low or video too short.")
