"""Bounded clipboard PNG input shared by the voice and remote LLM services."""

from __future__ import annotations

import base64
import binascii
import hashlib
import struct
import zlib

MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_IMAGE_SIDE = 2048
PNG_PREFIX = "data:image/png;base64,"
MAX_IMAGE_DATA_CHARS = len(PNG_PREFIX) + 4 * ((MAX_IMAGE_BYTES + 2) // 3)


def image_metadata(data_url: str) -> dict:
    """Validate an inline PNG and return diagnostics without exposing its payload.

    The browser converts clipboard images to PNG, with a 2048-pixel longest side.
    Only inline data is accepted: neither service fetches user-supplied URLs.
    """
    if not isinstance(data_url, str) or not data_url.startswith(PNG_PREFIX):
        raise ValueError("Paste a PNG image from the clipboard.")
    if len(data_url) > MAX_IMAGE_DATA_CHARS:
        raise ValueError("The image exceeds the 4 MiB limit.")
    try:
        raw = base64.b64decode(data_url[len(PNG_PREFIX):], validate=True)
    except (ValueError, binascii.Error):
        raise ValueError("The clipboard image is invalid.") from None
    if len(raw) > MAX_IMAGE_BYTES or not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("The clipboard image is invalid or too large.")
    offset, width, height, image_data = 8, 0, 0, False
    bit_depth = color_type = interlace = None
    compressed = bytearray()
    while offset + 12 <= len(raw):
        size = struct.unpack_from("!I", raw, offset)[0]
        kind = raw[offset + 4:offset + 8]
        end = offset + 8 + size
        if end + 4 > len(raw):
            break
        if zlib.crc32(raw[offset + 4:end]) != struct.unpack_from("!I", raw, end)[0]:
            break
        if offset == 8:
            if kind != b"IHDR" or size != 13:
                break
            width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack_from("!IIBBBBB", raw, offset + 8)
            if not (0 < width <= MAX_IMAGE_SIDE and 0 < height <= MAX_IMAGE_SIDE):
                raise ValueError("The image must be at most 2048 pixels per side.")
            channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(color_type)
            if bit_depth != 8 or channels is None or compression != 0 or filtering != 0 or interlace != 0:
                raise ValueError("The clipboard PNG encoding is not supported.")
        if kind == b"IDAT" and size:
            image_data = True
            compressed.extend(raw[offset + 8:end])
        if kind == b"IEND":
            if size == 0 and end + 4 == len(raw) and image_data:
                expected = height * (1 + width * channels)
                try:
                    decoder = zlib.decompressobj()
                    pixels = decoder.decompress(compressed, expected + 1)
                except zlib.error:
                    raise ValueError("The clipboard PNG pixel data is invalid.") from None
                if len(pixels) != expected or decoder.unconsumed_tail or not decoder.eof:
                    raise ValueError("The clipboard PNG pixel data is invalid.")
                stride = width * channels + 1
                if any(pixels[row * stride] > 4 for row in range(height)):
                    raise ValueError("The clipboard PNG pixel data is invalid.")
                return {"mime_type": "image/png", "bytes": len(raw), "width": width,
                        "height": height, "sha256": hashlib.sha256(raw).hexdigest()}
            break
        offset = end + 4
    raise ValueError("The clipboard image is incomplete or invalid.")
