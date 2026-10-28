"""One short recording per low-priority CPU process; audio is never saved."""
import argparse
import io
import json
import os
import sys
import wave

from .voice import MAX_AUDIO_BYTES, validate_audio


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    args = p.parse_args()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    if hasattr(os, "nice"):
        os.nice(5)
    from vosk import Model, KaldiRecognizer, SetLogLevel
    SetLogLevel(-1)
    data = sys.stdin.buffer.read(MAX_AUDIO_BYTES + 1)
    validate_audio(data)
    with wave.open(io.BytesIO(data), "rb") as audio:
        recognizer = KaldiRecognizer(Model(args.model), 16000)
        parts = []
        while chunk := audio.readframes(4000):
            if recognizer.AcceptWaveform(chunk):
                parts.append(json.loads(recognizer.Result()).get("text", ""))
        parts.append(json.loads(recognizer.FinalResult()).get("text", ""))
    print(json.dumps(dict(text=" ".join(filter(None, parts))), ensure_ascii=False))


if __name__ == "__main__":
    main()
