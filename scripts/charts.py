"""Charts for the Trailgate evaluation: reliability, the skip-threshold trade-off, day against night."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parent.parent / "results"
BLUE, ORANGE, VIOLET, GREY = "#2a78d6", "#eb6834", "#6250d6", "#9a9893"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
LIGHT = {"day": BLUE, "night_ir": VIOLET}
LIGHT_NAME = {"day": "Day", "night_ir": "Night (infrared)"}


def style(ax, title: str, subtitle: str = "") -> None:
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold", pad=22 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=INK2)


def save(fig, name: str) -> None:
    fig.savefig(RESULTS / name, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def reliability_chart(metrics: dict) -> None:
    fig, (ax, axn) = plt.subplots(2, 1, figsize=(6.4, 6.2), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    fig.patch.set_facecolor("white")
    ax.plot([0, 1], [0, 1], color=GREY, linewidth=1, linestyle="--", label="Perfect calibration")
    allb = [b for b in metrics["reliability"] if b["n"]]
    ax.plot([b["mean_present"] for b in allb], [b["observed_animal_rate"] for b in allb], color=BLUE, linewidth=2,
            marker="o", markersize=8, markeredgecolor="white", markeredgewidth=1.5, label="Decider, held-out frames")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("Observed share of frames with an animal", color=INK2, fontsize=9)
    style(ax, "Decider P(animal) against what was in the frame",
          f"Held-out set, animal and empty frames ({sum(b['n'] for b in allb)}), 10 equal-width bins")
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK2)
    centers = [(b["lo"] + b["hi"]) / 2 for b in metrics["reliability"]]
    axn.bar(centers, [b["n"] - b["n_animal"] for b in metrics["reliability"]], width=0.08, color=GREY, label="Empty")
    axn.bar(centers, [b["n_animal"] for b in metrics["reliability"]], width=0.08,
            bottom=[b["n"] - b["n_animal"] for b in metrics["reliability"]], color=BLUE, label="Animal",
            edgecolor="white", linewidth=1)
    for c, b in zip(centers, metrics["reliability"]):
        if b["n"]:
            axn.text(c, b["n"] + 1, str(b["n"]), ha="center", fontsize=8, color=INK)
    axn.set_ylim(0, max(b["n"] for b in metrics["reliability"]) * 1.25)
    style(axn, "")
    axn.set_ylabel("Frames", color=INK2, fontsize=9)
    axn.set_xlabel("Decider P(animal)", color=INK2, fontsize=9)
    axn.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK2, ncol=2)
    save(fig, "reliability.png")


def tradeoff_chart(metrics: dict) -> None:
    pol = metrics["policies"]
    n_animal = pol["v1"]["heldout"]["all"]["n_animal"]
    n_frames = pol["v1"]["heldout"]["all"]["n_frames"]
    empty_share = pol["v1"]["heldout"]["all"]["n_empty"] / n_frames * 100
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    fig.patch.set_facecolor("white")
    top = 0
    specs = (("v1", GREY, "v1: per frame", metrics["thresholds"]["v1"]["skip_below"], (4, 0.62)),
             ("v2", BLUE, "v2: max over the burst", metrics["thresholds"]["skip_below"], (4, 0.36)))
    for name, color, label, frozen, (tx, ty) in specs:
        sw = pol[name]["sweep"]
        x = [r["frames_skipped_pct"] for r in sw]
        y = [r["animals_missed"] for r in sw]
        top = max(top, max(y))
        ax.plot(x, y, color=color, linewidth=2, drawstyle="steps-post", label=label)
        f = min(sw, key=lambda r: abs(r["skip_below"] - frozen))
        ax.plot(f["frames_skipped_pct"], f["animals_missed"], "o", color=ORANGE if name == "v2" else INK2, markersize=9,
                markeredgecolor="white", markeredgewidth=2, zorder=5)
        ax.annotate(f"{name}, frozen skip_below={frozen}:\n{f['frames_skipped_pct']}% skipped, {f['animals_missed']} of {n_animal} animals missed",
                    (f["frames_skipped_pct"], f["animals_missed"]), xytext=(tx, top * ty if top else 1), textcoords="data",
                    fontsize=8.5, color=INK, arrowprops={"arrowstyle": "-", "color": INK2, "linewidth": 0.8})
    for r in pol["v2"]["sweep"]:
        if round(r["skip_below"] * 100) % 10 == 0 and 0.2 <= r["skip_below"] <= 0.9:
            ax.annotate(f"{r['skip_below']:.1f}", (r["frames_skipped_pct"], r["animals_missed"]), textcoords="offset points",
                        xytext=(4, -11), fontsize=7.5, color=BLUE)
    ax.axvline(empty_share, color=GREY, linewidth=1, linestyle="--")
    ax.text(empty_share + 0.8, top * 0.97, "share of frames\nthat are empty", fontsize=8, color=INK2, va="top")
    ax.set_xlabel("Held-out frames skipped, no model called (%)", color=INK2, fontsize=9)
    ax.set_ylabel(f"Animal frames skipped (of {n_animal})", color=INK2, fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=-max(1, top * 0.02))
    style(ax, "The safety knob: skip more frames, miss more animals",
          f"Held-out set, {n_frames} frames. Blue labels: v2 skip_below values. Other thresholds frozen per policy.")
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK2)
    save(fig, "tradeoff.png")


def day_night_chart(metrics: dict) -> None:
    pol = metrics["policies"]
    groups = [("Empty frames skipped", "empty_skipped", "n_empty"), ("Animal frames missed", "animals_missed", "n_animal")]
    series = [("v1", "day", BLUE, 0.45), ("v2", "day", BLUE, 1.0), ("v1", "night_ir", VIOLET, 0.45), ("v2", "night_ir", VIOLET, 1.0)]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    fig.patch.set_facecolor("white")
    width = 0.19
    for i, (name, light, color, alpha) in enumerate(series):
        s = pol[name]["heldout"][light]
        xs = [g + (i - 1.5) * (width + 0.015) for g in range(len(groups))]
        vals = [100 * s[a] / s[b] if s[b] else 0 for _, a, b in groups]
        ax.bar(xs, vals, width=width, color=color, alpha=alpha, label=f"{name}, {LIGHT_NAME[light]}")
        for x, v, (_, a, b) in zip(xs, vals, groups):
            ax.text(x, v + 1.5, f"{v:.0f}%\n{s[a]}/{s[b]}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=9, color=INK2)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of frames in the group", color=INK2, fontsize=9)
    style(ax, "Day against night (infrared): v1 and v2", "Held-out frames, thresholds frozen on the tuning set. Pale bars: v1.")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK2, ncol=2)
    save(fig, "day_night.png")


def cost_chart(metrics: dict) -> None:
    c = metrics["cost"]
    rows = [("Nova Pro on every frame", c["nova_pro_every_frame"], GREY),
            ("Trailgate v1", c["policies"]["v1"]["routed"], GREY),
            ("Trailgate v2", c["policies"]["v2"]["routed"], BLUE),
            ("Nova Lite on every frame", c["nova_lite_every_frame"], GREY)]
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    fig.patch.set_facecolor("white")
    ys = list(range(len(rows)))[::-1]
    ax.barh(ys, [r[1] for r in rows], color=[r[2] for r in rows], height=0.55)
    for y, (_, v, _) in zip(ys, rows):
        ax.text(v + max(r[1] for r in rows) * 0.01, y, f"${v:.2f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(ys, [r[0] for r in rows], fontsize=9, color=INK2)
    ax.set_xlim(0, max(r[1] for r in rows) * 1.15)
    ax.set_xlabel("Estimated $ per 10,000 frames (published token formula and prices)", color=INK2, fontsize=9)
    style(ax, "What 10,000 camera-trap frames cost", "Held-out mix; Decider runs locally at $0 per call.")
    ax.grid(axis="y", visible=False)
    save(fig, "cost.png")


def draw_all(metrics: dict, frames=None) -> None:
    reliability_chart(metrics)
    tradeoff_chart(metrics)
    day_night_chart(metrics)
    cost_chart(metrics)
