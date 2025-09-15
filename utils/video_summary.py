from moviepy.editor import VideoFileClip, concatenate_videoclips

def create_short_summary(video_path, timestamps):
    """
    Creates a short summarized video based on selected timestamps.

    Args:
        video_path (str): Path to the original video
        timestamps (list of tuples): List of (start_sec, end_sec) segments to include

    Returns:
        str: Path to the generated summarized video
    """
    clips = []
    video = VideoFileClip(video_path)

    for start, end in timestamps:
        clip = video.subclip(start, end)
        clips.append(clip)

    final_clip = concatenate_videoclips(clips)
    output_path = "assets/short_summary.mp4"
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

    return output_path
