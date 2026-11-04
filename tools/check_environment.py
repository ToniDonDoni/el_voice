"""Check actual native imports and model files without loading model packages."""

import argparse
import importlib
import json
import os
import platform
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-models", action="store_true")
    args = parser.parse_args()
    failures = []
    print(f"Platform: {platform.system()} {platform.machine()}")
    # Load Torch first so its bundled OpenBLAS is registered before native inference
    # packages can load another library with the same SONAME.
    for name in ("torch", "fastapi", "uvicorn", "websockets", "numpy", "av",
                 "ctranslate2", "faster_whisper", "llama_cpp", "huggingface_hub",
                 "httpx", "langchain", "langchain_openai", "ddgs", "tiktoken"):
        try:
            module = importlib.import_module(name)
            print(f"OK {name}: {getattr(module, '__version__', 'imported')}")
            if name == "torch" and module.version.cuda is not None:
                failures.append("Expected CPU-only PyTorch")
            if name == "tiktoken":
                try:
                    module.get_encoding("cl100k_base").encode("offline tokenizer check")
                    print("OK tiktoken encoding: cl100k_base available without runtime download")
                except Exception as exc:
                    failures.append(f"tiktoken offline encoding: {exc}")
        except Exception as exc:
            failures.append(f"{name}: {exc}")
    if not args.skip_models:
        manifest = json.loads((Path(__file__).parent / "models.json").read_text())
        model_root = Path(os.environ.get("EL_VOICE_MODELS", "/models"))
        for asset in manifest["assets"]:
            path = model_root / asset["path"]
            if not path.is_file() or path.stat().st_size == 0:
                failures.append(f"Missing model file: {asset['path']}")
            else:
                print(f"OK model file: {asset['path']}")
        print("This checks file presence only. It does not run model inference.")
    for failure in failures:
        print(f"FAIL {failure}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
