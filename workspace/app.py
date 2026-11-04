"""Temporary environment status server; replace with our voice application later."""

import argparse
import json
import os
import platform
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = b"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>EL Voice - environment ready</title>
<style>body{background:#111b25;color:#edf4f6;font:18px system-ui;max-width:700px;
margin:14vh auto;padding:24px}small{color:#8cdaca}h1{font-size:48px}p{line-height:1.6}
code{color:#8cdaca}a{color:#8cdaca}</style>
<small>EL VOICE / SETUP</small><h1>The environment is ready.</h1>
<p>The container can serve this page through the published local port.</p>
<p>The voice conversation pipeline is not implemented yet. This page does not record
your microphone or generate AI replies.</p>
<p>Application code lives in the mounted workspace. Model files are stored separately.</p>
<a href="/api/status">View environment status</a>
<h2>Voice test: b_lez</h2>
<p>After generating a test sample, listen to the selected voice here.</p>
<audio controls preload="none" src="/audio/tts-test.wav"></audio>
<p><a href="/audio/tts-test.wav">Open the generated WAV</a></p></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/healthz":
            content = json.dumps({"status": "ok", "stage": "bootstrap"}).encode()
            content_type = "application/json"
        elif self.path == "/api/status":
            root = Path(os.environ.get("EL_VOICE_MODELS", "/models"))
            content = json.dumps({
                "stage": "bootstrap", "voice_pipeline_ready": False,
                "system": platform.system(), "architecture": platform.machine(),
                "workspace_mounted": Path("/workspace/app.py").is_file(),
                "model_file_count": sum(1 for p in root.rglob("*")
                                        if p.is_file() and not p.name.startswith(".")),
            }).encode()
            content_type = "application/json"
        elif self.path in ("/", "/index.html"):
            content, content_type = PAGE, "text/html; charset=utf-8"
        elif self.path == "/audio/tts-test.wav":
            sample = Path(os.environ.get("EL_VOICE_DATA", "/data")) / "tts-test.wav"
            if not sample.is_file():
                self.send_error(404, "Run the voice synthesis test to generate a sample first")
                return
            content, content_type = sample.read_bytes(), "audio/wav"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; media-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()
    print(f"Bootstrap status server listening on {args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
