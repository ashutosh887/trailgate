"""Render docs/upload/cover.jpg, the text-free 1200x675 Builder Center cover: a card of
held-out frames flows into the Decider gate and splits into the four routes (skip, Nova Lite,
Nova Pro, alert). Route frames are the ones scripts/cover.py stamps with their real routes."""

from __future__ import annotations

import csv
import random
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
S = 2  # supersample, downscaled at the end
W, H = 1200 * S, 675 * S
GATE = (585 * S, 338 * S)

SKIP, LITE, PRO, ALERT = "#8b979f", "#6fdc93", "#f0c060", "#ff5f5f"
ROUTES = [  # (frame, colour, centre, width)
    ("586e30ae-23d2-11e8-a6a3-ec086b02610b.jpg", LITE, (990, 120), 280),
    ("59167994-23d2-11e8-a6a3-ec086b02610b.jpg", PRO, (990, 338), 280),
    ("5869356a-23d2-11e8-a6a3-ec086b02610b.jpg", ALERT, (990, 556), 280),
]
SKIPPED = ["598ad04d-23d2-11e8-a6a3-ec086b02610b.jpg", "5a096c21-23d2-11e8-a6a3-ec086b02610b.jpg"]


def rgba(hex_colour: str, alpha: int) -> tuple[int, int, int, int]:
    h = hex_colour.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha)


def bezier(p0, p1, p2, p3, n=120):
    for i in range(n + 1):
        t = i / n
        u = 1 - t
        yield (
            u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0],
            u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1],
        )


def thumb(name: str, w: int, ratio: float = 0.75) -> Image.Image:
    img = Image.open(ROOT / "data/images" / name).convert("RGB")
    h = int(w * ratio)
    scale = max(w / img.width, h / img.height)
    img = img.resize((int(img.width * scale) + 1, int(img.height * scale) + 1), Image.LANCZOS)
    left, top = (img.width - w) // 2, (img.height - h) // 2
    return img.crop((left, top, left + w, top + h))


def framed(img: Image.Image, colour: str, border: int) -> Image.Image:
    out = Image.new("RGBA", (img.width + border * 2, img.height + border * 2), rgba(colour, 255))
    out.paste(img, (border, border))
    return out


def glow(layer: Image.Image, box, colour: str, spread: int, alpha: int) -> None:
    g = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    ImageDraw.Draw(g).rounded_rectangle(box, radius=spread // 2, fill=rgba(colour, alpha))
    layer.alpha_composite(g.filter(ImageFilter.GaussianBlur(spread)))


def background() -> Image.Image:
    bg = Image.new("RGB", (W, H))
    px = ImageDraw.Draw(bg)
    for y in range(H):
        t = y / H
        px.line([(0, y), (W, y)], fill=(int(6 + 6 * t), int(14 + 10 * t), int(11 + 7 * t)))
    halo = Image.new("RGB", (W, H), (0, 0, 0))
    ImageDraw.Draw(halo).ellipse(
        (GATE[0] - 380 * S, GATE[1] - 300 * S, GATE[0] + 380 * S, GATE[1] + 300 * S), fill=(18, 44, 34)
    )
    bg = ImageChops.add(bg, halo.filter(ImageFilter.GaussianBlur(160 * S)))
    noise = Image.effect_noise((W, H), 18).convert("RGB")
    return Image.blend(bg, ImageChops.multiply(bg, noise), 0.12).convert("RGBA")


def card_frames(rng: random.Random) -> list[str]:
    rows = list(csv.DictReader((ROOT / "data/heldout.csv").open()))
    have = {p.name for p in (ROOT / "data/images").iterdir()}
    rows = [r for r in rows if r["file_name"] in have]
    empty = [r["file_name"] for r in rows if r["label"] == "empty"]
    animal = [r["file_name"] for r in rows if r["label"] not in ("empty", "car")]
    picks = rng.sample(empty, 9) + rng.sample(animal, 7)
    rng.shuffle(picks)
    return picks


def stream_mask(sources) -> Image.Image:
    # Keep streams off the card itself, so they read as leaving its right edge.
    mask = Image.new("L", (W, H), 0)
    edge = max(x for x, _ in sources) - 40 * S
    ImageDraw.Draw(mask).rectangle((edge, 0, W, H), fill=255)
    return mask.filter(ImageFilter.GaussianBlur(30 * S))


def main() -> None:
    rng = random.Random(887)
    canvas = background()

    # The night's card: a loose cascade of frames on the left, dimmed.
    sources = []
    for i, name in enumerate(card_frames(rng)):
        col, row = i % 4, i // 4
        w = rng.randint(118, 138) * S
        img = thumb(name, w)
        img = Image.blend(img, Image.new("RGB", img.size, (8, 16, 12)), 0.25 + 0.1 * (col == 0))
        tile = framed(img, "#22352c", 2 * S).rotate(rng.uniform(-7, 7), Image.BICUBIC, expand=True)
        cx = (95 + col * 108 + rng.randint(-14, 14) + row * 10) * S
        cy = (95 + row * 160 + rng.randint(-18, 18)) * S
        shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        ImageDraw.Draw(shadow).rectangle(
            (cx - tile.width // 2 + 8 * S, cy - tile.height // 2 + 12 * S, cx + tile.width // 2 + 8 * S, cy + tile.height // 2 + 12 * S),
            fill=(0, 0, 0, 150),
        )
        canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(10 * S)))
        canvas.alpha_composite(tile, (cx - tile.width // 2, cy - tile.height // 2))
        sources.append((cx + tile.width // 2 - 6 * S, cy))

    # Streams from the card into the gate.
    streams = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(streams)
    edge = max(x for x, _ in sources)
    for sy in sorted({y for _, y in sources}) * 2:
        sx, sy = edge - 30 * S, sy + rng.randint(-50, 50) * S
        pts = list(bezier((sx, sy), (sx + 90 * S, sy), (GATE[0] - 110 * S, GATE[1]), GATE))
        d.line(pts, fill=(150, 220, 185, 95), width=2 * S)
    blurred = streams.filter(ImageFilter.GaussianBlur(1.2 * S))
    canvas.alpha_composite(Image.composite(blurred, Image.new("RGBA", canvas.size, (0, 0, 0, 0)), stream_mask(sources)))

    # Routes out of the gate.
    routes = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(routes)
    targets = [(colour, (cx - w // 2) * S, cy * S) for _, colour, (cx, cy), w in ROUTES]
    skip_end = (690 * S, 585 * S)
    for colour, tx, ty in targets:
        pts = list(bezier(GATE, (GATE[0] + 120 * S, GATE[1]), (tx - 120 * S, ty), (tx, ty)))
        d.line(pts, fill=rgba(colour, 230), width=4 * S)
    skip_pts = list(bezier(GATE, (GATE[0] + 60 * S, GATE[1] + 40 * S), (skip_end[0] - 90 * S, skip_end[1] - 120 * S), skip_end))
    for i in range(0, len(skip_pts) - 6, 10):  # dashed: these frames reach no model
        d.line(skip_pts[i : i + 6], fill=rgba(SKIP, 170), width=3 * S)
    halo = routes.filter(ImageFilter.GaussianBlur(9 * S))
    canvas.alpha_composite(halo)
    canvas.alpha_composite(halo)
    canvas.alpha_composite(routes)

    # Skipped frames settle, faded, at the end of the dashed path.
    for k, name in enumerate(SKIPPED):
        img = thumb(name, 96 * S)
        img = Image.blend(img, Image.new("RGB", img.size, (8, 16, 12)), 0.6)
        tile = framed(img, SKIP, 2 * S).convert("RGBA")
        tile.putalpha(150)
        tile = tile.rotate(-10 + k * 14, Image.BICUBIC, expand=True)
        canvas.alpha_composite(tile, (skip_end[0] - (40 - k * 46) * S, skip_end[1] - (10 + k * 8) * S))

    # The three frames that reach a model or the ranger.
    for name, colour, (cx, cy), w in ROUTES:
        img = thumb(name, w * S, ratio=0.62)
        box = (cx * S - img.width // 2, cy * S - img.height // 2)
        glow(canvas, (box[0], box[1], box[0] + img.width, box[1] + img.height), colour, 26 * S, 150 if colour == ALERT else 105)
        canvas.alpha_composite(framed(img, colour, 4 * S), (box[0] - 4 * S, box[1] - 4 * S))

    # The Decider gate: a lit node with a ring.
    node = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(node)
    gx, gy = GATE
    d.ellipse((gx - 46 * S, gy - 46 * S, gx + 46 * S, gy + 46 * S), outline=(190, 255, 220, 200), width=3 * S)
    d.ellipse((gx - 20 * S, gy - 20 * S, gx + 20 * S, gy + 20 * S), fill=(225, 255, 238, 255))
    canvas.alpha_composite(node.filter(ImageFilter.GaussianBlur(22 * S)))
    canvas.alpha_composite(node.filter(ImageFilter.GaussianBlur(6 * S)))
    canvas.alpha_composite(node)

    out = ROOT / "docs/upload/cover.jpg"
    canvas.convert("RGB").resize((1200, 675), Image.LANCZOS).save(out, quality=90, optimize=True)
    print(out, out.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
