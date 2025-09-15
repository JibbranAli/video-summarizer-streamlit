# app.py
import streamlit as st
import tempfile
import os
import whisper
from transformers import pipeline
import ffmpeg
import cv2
from PIL import Image

# ----------------------------
# App Title
# ----------------------------
st.set_page_config(page_title="AI Video Summarizer", layout="wide")
st.title("🎬 AI-Powered Video Summarizer")
st.markdown(
    """
Upload your lecture, meeting, or long video and get:
- Full **transcription**
- **Short summary**
- Optional **keyframes preview**
"""
)

# ----------------------------
# Video Upload
# ----------------------------
video_file = st.file_uploader("Upload Video", type=["mp4","mov","mkv"])
if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    video_path = tfile.name
    st.video(video_path)

    st.info("✅ Video uploaded successfully!")

    # ----------------------------
    # Extract Audio using FFmpeg
    # ----------------------------
    audio_path = "audio.wav"
    try:
        ffmpeg.input(video_path).output(audio_path, ac=1, ar='16k').overwrite_output().run()
        st.success("🎧 Audio extracted successfully!")
    except Exception as e:
        st.error(f"Error extracting audio: {e}")

    # ----------------------------
    # Transcribe Audio using Whisper
    # ----------------------------
    st.info("⏳ Transcribing audio using Whisper...")
    model = whisper.load_model("base")  # Choose 'tiny', 'base', 'small', 'medium', 'large'
    result = model.transcribe(audio_path)
    transcript = result["text"]
    st.subheader("📝 Full Transcription")
    st.text_area("Transcript", transcript, height=300)

    # ----------------------------
    # Summarize Transcript
    # ----------------------------
    st.info("🤖 Summarizing transcript...")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(transcript, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
    st.subheader("📌 Video Summary")
    st.write(summary)

    # Download options
    st.download_button("📥 Download Summary", summary, file_name="summary.txt")
    st.download_button("📥 Download Transcript", transcript, file_name="transcript.txt")

    # ----------------------------
    # Optional: Keyframe Extraction
    # ----------------------------
    st.subheader("🖼 Keyframes Preview")
    cap = cv2.VideoCapture(video_path)
    frame_rate = 60  # extract 1 frame every 60 frames (~2 sec for 30fps)
    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_rate == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        count += 1
    cap.release()

    # Display frames
    cols = st.columns(4)
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame)
        cols[i % 4].image(img, use_column_width=True)
