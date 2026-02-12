def main():
    """
    Transcribe a simple audio file to text.
    """

    import whisper

    # Parameters
    whisper_model = "small.en"
    audio_file = "test_audio.wav"
    transcript_file = "test_transcript.txt"

    # Instantiate the speech recognition model
    model = whisper.load_model(whisper_model)

    # Transcribe
    result = model.transcribe(audio_file)

    # Output the result
    with open(transcript_file, "w") as f:
        f.write(result["text"])

if __name__ == "__main__":
    main()
