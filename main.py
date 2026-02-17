import argparse
from pathlib import Path

import whisper


def format_timestamp(seconds: float) -> str:
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes:02d}:{secs:02d}"


def main():
    """
    Transcribe an audio file to text using Whisper
    """

    parser = argparse.ArgumentParser(description="Transcribe an audio file to text using Whisper.")
    parser.add_argument("audio_file", type=Path, help="Path to the audio file to transcribe")
    parser.add_argument("--model", default="small.en", help="Whisper model to use (default: small.en)")
    parser.add_argument("--output", type=Path, help="Output text file (default: same name as audio with .txt extension)")
    parser.add_argument("--timestamps", action=argparse.BooleanOptionalAction, default=True, help="Include segment timestamps in output (default: True)")
    args = parser.parse_args()

    if not args.audio_file.exists():
        parser.error(f"Audio file not found: {args.audio_file}")

    available_models = whisper.available_models()
    if args.model not in available_models:
        parser.error(f"Unknown model '{args.model}'. Available: {', '.join(available_models)}")

    transcript_file = args.output or args.audio_file.with_suffix(".txt")

    output_dir = transcript_file.parent
    if not output_dir.is_dir():
        parser.error(f"Output directory does not exist: {output_dir}")

    # Instantiate the speech recognition model
    model = whisper.load_model(args.model)

    # Transcribe
    result = model.transcribe(str(args.audio_file))

    # Output the result
    with open(transcript_file, "w") as f:
        if args.timestamps:
            for segment in result["segments"]:
                start = format_timestamp(segment["start"])
                end = format_timestamp(segment["end"])
                text = segment["text"].strip()
                f.write(f"[{start} - {end}] {text}\n")
        else:
            f.write(result["text"])

if __name__ == "__main__":
    main()
