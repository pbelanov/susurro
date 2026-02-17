import argparse
from pathlib import Path

import whisper


def main():
    """
    Transcribe an audio file to text using Whisper
    """

    parser = argparse.ArgumentParser(description="Transcribe an audio file to text using Whisper.")
    parser.add_argument("audio_file", type=Path, help="Path to the audio file to transcribe")
    parser.add_argument("--model", default="small.en", help="Whisper model to use (default: small.en)")
    parser.add_argument("--output", type=Path, help="Output text file (default: same name as audio with .txt extension)")
    args = parser.parse_args()

    if not args.audio_file.exists():
        parser.error(f"Audio file not found: {args.audio_file}")

    transcript_file = args.output or args.audio_file.with_suffix(".txt")

    # Instantiate the speech recognition model
    model = whisper.load_model(args.model)

    # Transcribe
    result = model.transcribe(str(args.audio_file))

    # Output the result
    with open(transcript_file, "w") as f:
        f.write(result["text"])

if __name__ == "__main__":
    main()
