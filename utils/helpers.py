from transformers import pipeline

def get_important_timestamps(transcript, total_duration):
    """
    Simple AI logic to determine key video segments based on transcript.
    Returns a list of (start_sec, end_sec) tuples.

    Args:
        transcript (str): Full transcription text
        total_duration (float): Total video duration in seconds

    Returns:
        list: List of (start, end) seconds tuples
    """
    summarizer = pipeline("summarization")
    summary = summarizer(transcript, max_length=100, min_length=30, do_sample=False)[0]['summary_text']

    # Dummy logic: split video evenly for each summarized sentence
    sentences = summary.split(". ")
    segment_length = total_duration / len(sentences)
    timestamps = [(int(i * segment_length), int((i + 1) * segment_length)) for i in range(len(sentences))]
    return timestamps
