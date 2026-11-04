"""Generate a real CPU voice sample after checking the downloaded model receipt."""

import argparse
import hashlib
import json
import os
import time
import wave
from pathlib import Path

MODEL_URL = "https://models.silero.ai/models/tts/cyr/v4_cyrillic.pt"
MODEL_NAME = "tts/v4_cyrillic.pt"


def checked_model_path(model_root, relative_path):
    root = Path(model_root).resolve()
    if relative_path != MODEL_NAME:
        raise ValueError("This smoke check only loads the configured official Cyrillic model")
    model = (root / relative_path).resolve()
    model.relative_to(root)
    receipt_file = root / ".download-receipts.json"
    if not receipt_file.is_file():
        raise ValueError("Model download receipt missing; run the download service first")
    receipt = json.loads(receipt_file.read_text()).get(relative_path)
    if not receipt or receipt.get("url") != MODEL_URL:
        raise ValueError("Model receipt does not match the configured official publisher URL")
    digest = hashlib.sha256()
    with model.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != receipt.get("sha256"):
        raise ValueError("Model checksum differs from its download receipt; refusing to load it")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, help="A short sentence in the language to test")
    parser.add_argument("--output", type=Path, default=Path("/data/tts-test.wav"))
    args = parser.parse_args()
    if not args.text.strip() or len(args.text) > 500:
        parser.error("Provide between 1 and 500 characters of text")
    settings = json.loads(Path(__file__).with_name("settings.json").read_text())["tts"]
    model_path = checked_model_path(os.environ.get("EL_VOICE_MODELS", "/models"), settings["model"])

    import torch

    torch.set_num_threads(settings["cpu_threads"])
    started = time.perf_counter()
    model = torch.package.PackageImporter(str(model_path)).load_pickle("tts_models", "model")
    model.to(torch.device("cpu"))
    load_seconds = time.perf_counter() - started
    speaker = settings["speaker"]
    if speaker not in model.speakers:
        raise ValueError(f"Configured speaker is absent from this model: {speaker}")
    sample_rate = settings["sample_rate"]
    started = time.perf_counter()
    with torch.inference_mode():
        audio = model.apply_tts(text=args.text, speaker=speaker, sample_rate=sample_rate)
    generation_seconds = time.perf_counter() - started
    audio = audio.detach().cpu().float().flatten()
    if audio.numel() == 0 or not bool(torch.isfinite(audio).all()):
        raise ValueError("Synthesis returned an empty or non-finite waveform")
    if float(audio.abs().max()) < 0.00001:
        raise ValueError("Synthesis returned a silent waveform")
    pcm = (audio.clamp(-1, 1) * 32767).to(torch.int16).numpy().astype("<i2", copy=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.output), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())
    duration = audio.numel() / sample_rate
    with wave.open(str(args.output), "rb") as check:
        assert check.getnframes() > 0 and check.getframerate() == sample_rate
    print(f"PASS: speaker={speaker}, sample_rate={sample_rate}, device=cpu")
    print(f"Model loading: {load_seconds:.2f}s; generation: {generation_seconds:.2f}s")
    print(f"Audio duration: {duration:.2f}s; real-time factor: {generation_seconds / duration:.2f}")
    print(f"WAV saved: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(f"TTS test failed: {error}") from error
