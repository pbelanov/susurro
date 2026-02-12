def main():
    import whisper

    model = whisper.load_model("small.en")
    result = model.transcribe("testaudio.wav")
    print(result["text"])

if __name__ == "__main__":
    main()
