"""Small synthetic image fixture; no screenshots or user data are stored."""

import base64
import struct
import zlib


def shapes_png(width=320, height=180):
    def chunk(kind, data):
        return struct.pack("!I", len(data)) + kind + data + struct.pack("!I", zlib.crc32(kind + data))

    rows = []
    for y in range(height):
        row = bytearray()
        for x in range(width):
            color = (255, 255, 255)
            if 20 < x < 120 and 40 < y < 140:
                color = (255, 0, 0)
            if (x - 240) ** 2 + (y - 90) ** 2 < 50 ** 2:
                color = (0, 0, 255)
            row.extend(color)
        rows.append(b"\x00" + row)
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"".join(rows))) + chunk(b"IEND", b""))


def image_url():
    return "data:image/png;base64," + base64.b64encode(shapes_png()).decode()
