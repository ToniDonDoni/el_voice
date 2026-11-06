"""Fetch configured model assets without importing or executing model code."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

ALLOWED_HOSTS = {"huggingface.co", "models.silero.ai", "raw.githubusercontent.com", "www.modelscope.cn"}


def get_manifest():
    beside_script = Path(__file__).with_name("models.json")
    path = beside_script if beside_script.exists() else Path(__file__).resolve().parents[1] / "models.json"
    return json.loads(path.read_text())["assets"]


def fingerprint(path):
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1()
    git_blob.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha256.update(block)
            git_blob.update(block)
    return sha256.hexdigest(), git_blob.hexdigest()


def validate(path, asset, receipt=None):
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty file: {asset['path']}")
    if asset.get("size") is not None and path.stat().st_size != asset["size"]:
        raise ValueError(f"Size mismatch: {asset['path']}")
    sha256, git_blob = fingerprint(path)
    if asset.get("sha256") and sha256 != asset["sha256"]:
        raise ValueError(f"SHA256 mismatch: {asset['path']}")
    if asset.get("git_blob_sha1") and git_blob != asset["git_blob_sha1"]:
        raise ValueError(f"Git blob checksum mismatch: {asset['path']}")
    if receipt:
        if receipt["sha256"] != sha256 or receipt["url"] != asset["url"]:
            raise ValueError(f"Local receipt mismatch: {asset['path']}")
    return {"url": asset["url"], "sha256": sha256, "size": path.stat().st_size}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("EL_VOICE_MODELS", "/models")))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--only", nargs="+", help="limit verification/download to matching paths or directories")
    args = parser.parse_args()
    assets = get_manifest()
    if args.only:
        assets = [asset for asset in assets if any(
            asset["path"] == item or asset["path"].startswith(item.rstrip("/") + "/")
            for item in args.only
        )]
        if not assets:
            raise ValueError("--only did not match any configured model assets")
    if args.list:
        for asset in assets:
            print(f"{asset['path']}\n  {asset['url']}")
        return
    root = args.root.resolve()
    if not args.verify:
        root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / ".download-receipts.json"
    receipts = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    for asset in assets:
        relative = PurePosixPath(asset["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Model path must stay inside the models directory")
        url = urlparse(asset["url"])
        if url.scheme != "https" or url.hostname not in ALLOWED_HOSTS:
            raise ValueError("Unexpected model publisher")
        target = (root / asset["path"]).resolve()
        target.relative_to(root)
        receipt = receipts.get(asset["path"])
        has_publisher_hash = bool(asset.get("sha256") or asset.get("git_blob_sha1"))
        if args.verify:
            if not has_publisher_hash and not receipt:
                raise ValueError(f"No local download receipt for {asset['path']}")
            validate(target, asset, receipt)
            print(f"Verified: {asset['path']}", flush=True)
            continue
        if target.exists():
            if not has_publisher_hash and not receipt:
                raise ValueError(f"Refusing an existing unverified package: {asset['path']}")
            receipts[asset["path"]] = validate(target, asset, receipt)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".part")
            partial.resolve().relative_to(root)
            if not has_publisher_hash:
                print(f"Official HTTPS source, first-download trust: {asset['path']}", flush=True)
            print(f"Downloading: {asset['path']}", flush=True)
            subprocess.run([
                "curl", "--fail", "--location", "--show-error", "--retry", "5",
                "--connect-timeout", "30", "--speed-limit", "1024", "--speed-time", "120",
                "--proto", "=https", "--proto-redir", "=https", "--continue-at", "-",
                "--output", str(partial), asset["url"],
            ], check=True)
            try:
                receipts[asset["path"]] = validate(partial, asset)
            except ValueError:
                partial.unlink(missing_ok=True)
                raise
            partial.replace(target)
        temporary = receipt_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(receipts, indent=2) + "\n")
        temporary.replace(receipt_path)
        print(f"Ready: {asset['path']}", flush=True)
    print("Model files checked. No model code was loaded or executed.", flush=True)


if __name__ == "__main__":
    main()
