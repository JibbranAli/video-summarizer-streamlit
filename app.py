import streamlit as st
import whisper
from moviepy.editor import VideoFileClip
from utils.video_summary import create_short_summary
from utils.helpers import get_important_timestamps
import os

# Ensure assets folder exists
os.makedirs("assets", exist_ok=True)

st.set_page_config(page_title="AI Video Summarizer", layout="wide")
st.title("🎬 AI Video Summarizer & Short Video Creator")

# Upload video
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "mkv"])

if uploaded_file:
    video_path = f"assets/{uploaded_file.name}"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.video(video_path)

    # Get video duration
    video_clip = VideoFileClip(video_path)
    duration = video_clip.duration

    st.info("⏳ Transcribing audio with Whisper...")
    model = whisper.load_model("base")
    result = model.transcribe(video_path)
    transcript = result["text"]

    st.subheader("📝 Full Transcript")
    st.text_area("Transcript", transcript, height=300)

    # AI decides important segments
    st.info("🤖 AI analyzing important segments for summary...")
    timestamps = get_important_timestamps(transcript, duration)

    # Generate short video
    st.info("⏳ Creating AI short video summary...")
    summary_path = create_short_summary(video_path, timestamps)

    st.subheader("🎞 Short AI Video Summary")
    st.video(summary_path)
    st.success("✅ Short video summary generated successfully!"