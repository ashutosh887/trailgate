"""Code checks that run before Decider sees a frame.

Decider stays confident on a blank image (model card), so frames that are
missing, undecodable, black, blown out or a single flat colour are caught here.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageStat, UnidentifiedImageError

MIN_SIDE = 64
DARK_MEAN = 6.0
BRIGHT_MEAN = 250.0
FLAT_STDDEV = 3.0
IR_CHANNEL_SPREAD = 4.0


@dataclass(frozen=True)
class FrameCheck:
    ok: bool
    reason: str | None
    width: int = 0
    height: int = 0
    infrared: bool = False


def load(image: bytes | str | Path) -> bytes:
    return Path(image).read_bytes() if isinstance(image, (str, Path)) else image


def open_image(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img.load()
    return img.convert("RGB")


def is_infrared(img: Image.Image) -> bool:
    raw = img.convert("RGB").resize((64, 48)).tobytes()
    spread = [max(p) - min(p) for p in zip(raw[0::3], raw[1::3], raw[2::3], strict=True)]
    return sum(spread) / len(spread) < IR_CHANNEL_SPREAD


def check(image: bytes | str | Path) -> FrameCheck:
    try:
        data = load(image)
    except OSError as exc:
        return FrameCheck(False, f"unreadable: {exc.strerror or exc}")
    if not data:
        return FrameCheck(False, "unreadable: empty file")
    try:
        img = open_image(data)
    except UnidentifiedImageError:
        return FrameCheck(False, "undecodable: not a supported image format")
    except (OSError, ValueError) as exc:
        return FrameCheck(False, f"undecodable: {exc}")
    width, height = img.size
    if min(width, height) < MIN_SIDE:
        return FrameCheck(False, f"too small: {width}x{height}", width, height)
    stat = ImageStat.Stat(img.convert("L").resize((128, 96)))
    mean, stddev = stat.mean[0], stat.stddev[0]
    if mean < DARK_MEAN:
        return FrameCheck(False, f"near-black frame (mean {mean:.1f})", width, height)
    if mean > BRIGHT_MEAN:
        return FrameCheck(False, f"blown-out frame (mean {mean:.1f})", width, height)
    if stddev < FLAT_STDDEV:
        return FrameCheck(False, f"near-uniform frame (stddev {stddev:.1f})", width, height)
    return FrameCheck(True, None, width, height, is_infrared(img))


def image_format(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    return "jpeg"
