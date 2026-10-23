"""Render docs/images/cover.png: six held-out Caltech Camera Traps frames, each stamped with
the route Trailgate gave it and the Decider probability behind that route (results/decider_answers.jsonl)."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FRAMES = [
    ("598ad04d-23d2-11e8-a6a3-ec086b02610b.jpg", "SKIP", "no model · $0", "present", "#9aa5ad"),
    ("586e30ae-23d2-11e8-a6a3-ec086b02610b.jpg", "NOVA LITE", "clear animal", "present", "#7fd99a"),
    ("59167994-23d2-11e8-a6a3-ec086b02610b.jpg", "NOVA PRO", "Decider unsure · night IR", "present", "#e5c07b"),
    ("5869356a-23d2-11e8-a6a3-ec086b02610b.jpg", "ALERT", "vehicle on the trail", "escalate", "#ff6b6b"),
    ("58ce4099-23d2-11e8-a6a3-ec086b02610b.jpg", "ALERT", "person · CCT labels it empty", "escalate", "#ff6b6b"),
    ("5a096c21-23d2-11e8-a6a3-ec086b02610b.jpg", "SKIP", "no model · $0", "present", "#9aa5ad"),
]
TILE_W, TILE_H, GAP, PAD, HEAD = 640, 467, 16, 48, 150
BG = "#0d1512"


def font(size: int, mono: bool = False) -> ImageFont.FreeTypeFont:
    path = "/System/Library/Fonts/Menlo.ttc" if mono else "/System/Library/Fonts/HelveticaNeue.ttc"
    return ImageFont.truetype(path, size, index=1 if not mono else 0)


def main() -> None:
    answers = {}
    for line in (ROOT / "results/decider_answers.jsonl").read_text().splitlines():
        row = json.loads(line)
        answers[row["file_name"]] = row["answers"]

    width = PAD * 2 + TILE_W * 3 + GAP * 2
    height = HEAD + PAD + TILE_H * 2 + GAP
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((PAD, 40), "Trailgate", font=font(52), fill="#e8efe9")
    draw.text(
        (PAD, 104),
        "A 2B model decides which camera-trap photos are worth a model's time.",
        font=font(26),
        fill="#93a59b",
    )

    for i, (name, stamp, note, key, color) in enumerate(FRAMES):
        x = PAD + (i % 3) * (TILE_W + GAP)
        y = HEAD + (i // 3) * (TILE_H + GAP)
        img = Image.open(ROOT / "data/images" / name).convert("RGB")
        img = img.resize((TILE_W, int(img.height * TILE_W / img.width))).crop((0, 0, TILE_W, TILE_H))
        if stamp == "SKIP":
            img = Image.blend(img, Image.new("RGB", img.size, BG), 0.45)
        canvas.paste(img, (x, y))
        draw.rectangle((x, y, x + TILE_W - 1, y + TILE_H - 1), outline=color, width=4)
        p = answers[name][key]["noul"]
        label = f"P({'animal' if key == 'present' else 'person/vehicle'}) {p:.2f}"
        draw.rectangle((x, y + TILE_H - 84, x + TILE_W, y + TILE_H), fill=(10, 16, 14))
        draw.text((x + 20, y + TILE_H - 74), stamp, font=font(28), fill=color)
        draw.text((x + 20, y + TILE_H - 36), note, font=font(20), fill="#c9d3cc")
        draw.text((x + TILE_W - 20, y + TILE_H - 68), label, font=font(20, mono=True), fill="#e8efe9", anchor="ra")

    out = ROOT / "docs/images/cover.png"
    canvas.save(out)
    print(out)


if __name__ == "__main__":
    main()
