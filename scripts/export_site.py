"""Export the ranger console as a static site in site/: replay data as JSON, resized frames, the same web files.

The static page replays the real recorded Decider answers (results/decider_answers.jsonl) routed with the v2
burst policy, exactly as `python -m trailgate.server` does in replay mode. Nothing calls Decider or Bedrock.

Run:
  python scripts/export_site.py
  python scripts/export_site.py --site-url https://trailgate.vercel.app
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image

from trailgate import server

SITE = ROOT / "site"
REPO = "https://github.com/ashutosh887/trailgate"
TITLE = "Trailgate: camera-trap triage with Strands Decider 2B"
DESCRIPTION = (
    "A 2B model decides which camera-trap photos are worth a model's time: empty frames skipped with no model call, "
    "animals sent to the right-sized Amazon Nova model, people and vehicles straight to the ranger. "
    "Replay of real Strands Decider 2B answers on held-out Caltech Camera Traps frames."
)
THUMB_PX = 320
FULL_PX = 1024
FULL_QUALITY = 82
THUMB_QUALITY = 78
OG_WIDTH = 1200

ATTRIBUTION = """# Attribution

The camera-trap images in `frames/` and `thumbs/` are from **Caltech Camera Traps** (CCT):

> Sara Beery, Grant Van Horn, Pietro Perona. *Recognition in Terra Incognita.* ECCV 2018.
> https://lila.science/datasets/caltech-camera-traps

They are distributed under the **Community Data License Agreement, Permissive, Version 1.0** (CDLA-Permissive-1.0),
https://cdla.dev/permissive-1-0/. The images here are resized (thumbnails to {thumb}px, full frames to at most
{full}px on the long side, JPEG quality {q}); they are otherwise unchanged.

The Decider answers, routes and labels shown on this site come from the Trailgate repository,
{repo}, licensed Apache-2.0. Bedrock (Amazon Nova) costs are estimates from list prices and Nova's documented image
token counts; no Bedrock call is made by this site.
"""

VERCEL = {
    "cleanUrls": True,
    "trailingSlash": False,
    "headers": [
        {
            "source": "/(frames|thumbs)/(.*)",
            "headers": [{"key": "Cache-Control", "value": "public, max-age=31536000, immutable"}],
        },
        {
            "source": "/(og.jpg|favicon.svg)",
            "headers": [{"key": "Cache-Control", "value": "public, max-age=86400"}],
        },
        {
            "source": "/data/(.*)",
            "headers": [{"key": "Cache-Control", "value": "public, max-age=300, must-revalidate"}],
        },
        {
            "source": "/(.*)",
            "headers": [
                {"key": "X-Content-Type-Options", "value": "nosniff"},
                {"key": "Referrer-Policy", "value": "strict-origin-when-cross-origin"},
            ],
        },
    ],
}


def rewrite_urls(value):
    """Server image routes become relative static paths."""
    if isinstance(value, dict):
        return {k: rewrite_urls(v) for k, v in value.items()}
    if isinstance(value, list):
        return [rewrite_urls(v) for v in value]
    if isinstance(value, str):
        if value.startswith("/thumb/"):
            return "thumbs/" + value[len("/thumb/") :]
        if value.startswith("/frames/"):
            return "frames/" + value[len("/frames/") :]
    return value


def export_reel(reel: dict) -> tuple[dict, set[str], set[str]]:
    """One card as the server would stream it in replay mode at pace 1.0, plus the frames it needs."""
    frames, cache, source = server._resolve_reel(reel["id"], "replay")
    rows = []
    for meta in frames:
        result = server.replay_result(meta["file_name"], cache)
        row = server._enrich(result, meta, source)
        row["delay_s"] = round(server._replay_delay(result, 1.0), 4)
        rows.append(row)
    digest = server._digest(rows)
    full = {m["file_name"] for m in frames}
    thumbs = set(full)
    for meta in frames:
        thumbs.update(meta["burst"])
    for row in rows:
        thumbs.update(b["file_name"] for b in row["result"].get("burst") or [])
    data = {"id": reel["id"], "mode": "replay", "source": source, "frames": frames, "rows": rows, "digest": digest}
    return rewrite_urls(data), full, thumbs


def save_jpeg(src: Path, dest: Path, longest: int, quality: int) -> None:
    with Image.open(src) as img:
        img = img.convert("RGB")
        img.thumbnail((longest, longest))
        img.save(dest, "JPEG", quality=quality, optimize=True, progressive=True)


def head_meta(site_url: str, og_size: tuple[int, int]) -> str:
    base = site_url.rstrip("/")
    tags = [
        f"<title>{TITLE}</title>",
        f'<meta name="description" content="{DESCRIPTION}">',
        f'<link rel="canonical" href="{base}/">',
        '<link rel="icon" href="/favicon.svg" type="image/svg+xml">',
        '<meta property="og:type" content="website">',
        '<meta property="og:site_name" content="Trailgate">',
        f'<meta property="og:title" content="{TITLE}">',
        f'<meta property="og:description" content="{DESCRIPTION}">',
        f'<meta property="og:url" content="{base}/">',
        f'<meta property="og:image" content="{base}/og.jpg">',
        f'<meta property="og:image:width" content="{og_size[0]}">',
        f'<meta property="og:image:height" content="{og_size[1]}">',
        '<meta property="og:image:alt" content="Six held-out Caltech Camera Traps frames, each stamped with its route">',
        '<meta name="twitter:card" content="summary_large_image">',
        f'<meta name="twitter:title" content="{TITLE}">',
        f'<meta name="twitter:description" content="{DESCRIPTION}">',
        f'<meta name="twitter:image" content="{base}/og.jpg">',
    ]
    return "\n".join(tags)


def build_index(site_url: str, og_size: tuple[int, int]) -> str:
    """The console's index.html with full metadata and the static-mode flag; the body is unchanged."""
    html = (server.WEB / "index.html").read_text()
    html = _drop_line(html, '<meta name="description" content="A 2B')
    html = _drop_line(html, '<link rel="icon" href="/static/favicon.svg"')
    html = _replace_once(html, "<title>Trailgate</title>", head_meta(site_url, og_size))
    return _replace_once(
        html,
        '<script src="/static/app.js"></script>',
        '<script>window.TRAILGATE_STATIC = true;</script>\n<script src="/static/app.js"></script>',
    )


def _drop_line(text: str, prefix: str) -> str:
    start = text.index(prefix)
    return text[:start] + text[text.index("\n", start) + 1 :]


def _replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f"index.html: expected exactly one {old!r}")
    return text.replace(old, new)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the Trailgate console as a static site")
    parser.add_argument("--out", type=Path, default=SITE)
    parser.add_argument("--site-url", default="https://trailgate.vercel.app", help="absolute URL for Open Graph tags")
    args = parser.parse_args()
    out: Path = args.out

    for sub in ("data", "frames", "thumbs", "static"):
        shutil.rmtree(out / sub, ignore_errors=True)
    for sub in ("data", "frames", "thumbs"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    info = server.status_info(probe_decider=False)
    status = {
        "mode": "replay",
        "static": True,
        "repo": REPO,
        "record": None,
        "decider": {"reachable": False},
        "pipeline": {"available": False},
        "replay": {"frames": info["replay"]["frames"]},
        "reels": info["reels"],
        "questions": info["questions"],
        "thresholds": info["thresholds"],
        "difficulty_note": info["difficulty_note"],
    }

    full: set[str] = set()
    thumbs: set[str] = set()
    for reel in server.REELS:
        data, f, t = export_reel(reel)
        if not data["frames"]:
            raise SystemExit(f"{reel['id']}: no frames; are data/images and results/decider_answers.jsonl present?")
        full |= f
        thumbs |= t
        (out / "data" / f"{reel['id']}.json").write_text(json.dumps(data, separators=(",", ":")))
        d = data["digest"]
        print(
            f"{reel['id']}: {d['frames']} frames, {d['empty']} empty, {d['lite']} lite, {d['pro']} pro, {d['alerts']} alerts"
        )
    (out / "data" / "status.json").write_text(json.dumps(status, indent=1))

    thumbs = {n for n in thumbs if (server.IMAGES / n).exists()}
    for name in sorted(full):
        save_jpeg(server.IMAGES / name, out / "frames" / name, FULL_PX, FULL_QUALITY)
    for name in sorted(thumbs):
        save_jpeg(server.IMAGES / name, out / "thumbs" / name, THUMB_PX, THUMB_QUALITY)

    shutil.copytree(server.WEB, out / "static", ignore=shutil.ignore_patterns("index.html"))
    (out / "favicon.svg").write_text((server.WEB / "favicon.svg").read_text())
    with Image.open(ROOT / "assets" / "cover.png") as img:
        img = img.convert("RGB")
        img.thumbnail((OG_WIDTH, OG_WIDTH))
        img.save(out / "og.jpg", "JPEG", quality=85, optimize=True, progressive=True)
        og_size = img.size
    (out / "index.html").write_text(build_index(args.site_url, og_size))
    (out / "vercel.json").write_text(json.dumps(VERCEL, indent=2) + "\n")
    (out / "ATTRIBUTION.md").write_text(
        ATTRIBUTION.format(thumb=THUMB_PX, full=FULL_PX, q=FULL_QUALITY, repo=REPO),
    )

    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"{len(full)} full frames, {len(thumbs)} thumbnails, {size / 1e6:.1f} MB in {out}")


if __name__ == "__main__":
    main()
