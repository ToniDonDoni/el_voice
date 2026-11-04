"""Run the internal LLM service: ``python -m llm_service``."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace

from .config import load_config
from .server import create_app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EL Voice internal LLM service")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = load_config()
    if args.host:
        config = replace(config, host=args.host)
    if args.port:
        config = replace(config, port=args.port)

    import uvicorn

    uvicorn.run(
        create_app(config=config),
        host=config.host,
        port=config.port,
        log_level="info",
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
