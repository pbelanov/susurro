def main():
    """
    Transcribe a simple audio file to text.
    """

    import whisper

    # Parameters
    whisper_model = "small.en"
    project_name = "whisper-test2"
    audio_extension = ".m4a"
    audio_file = project_name + audio_extension
    transcript_file = project_name + ".txt"

    # Instantiate the speech recognition model
    model = whisper.load_model(whisper_model)

    # Transcribe
    result = model.transcribe(audio_file)

    # Output the result
    with open(transcript_file, "w") as f:
        f.write(result["text"])

if __name__ == "__main__":
    main()
