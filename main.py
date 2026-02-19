import argparse
from pathlib import Path

from faster_whisper import WhisperModel


def format_timestamp(seconds: float) -> str:
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes:02d}:{secs:02d}"


def main():
    """
    Transcribe (or translate) an audio file to text using faster-whisper
    """

    parser = argparse.ArgumentParser(description="Transcribe an audio file to text using Whisper.")
    parser.add_argument("audio_file", type=Path, help="Path to the audio file to transcribe")
    parser.add_argument("--model", default="small", help="Whisper model to use (default: small)")
    parser.add_argument("--output", type=Path, help="Output text file (default: same name as audio with .txt extension)")
    parser.add_argument("--timestamps", action=argparse.BooleanOptionalAction, default=False, help="Include segment timestamps in output (default: False)")
    parser.add_argument("--translate", action="store_true", help="Translate to English instead of transcribing")
    args = parser.parse_args()

    if not args.audio_file.exists():
        parser.error(f"Audio file not found: {args.audio_file}")

    transcript_file = args.output or args.audio_file.with_suffix(".txt")

    output_dir = transcript_file.parent
    if not output_dir.is_dir():
        parser.error(f"Output directory does not exist: {output_dir}")

    task = "translate" if args.translate else "transcribe"

    # Instantiate the speech recognition model (CPU, int8 for efficiency)
    model = WhisperModel(args.model, device="cpu", compute_type="int8")

    # Transcribe / translate
    segments, info = model.transcribe(str(args.audio_file), task=task)

    print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

    # Output the result
    with open(transcript_file, "w") as f:
        for segment in segments:
            if args.timestamps:
                start = format_timestamp(segment.start)
                end = format_timestamp(segment.end)
                f.write(f"[{start} - {end}] {segment.text.strip()}\n")
            else:
                f.write(segment.text.strip() + "\n")

    print(f"Output written to: {transcript_file}")

if __name__ == "__main__":
    main()
