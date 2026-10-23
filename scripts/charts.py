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
    sweep = metrics["sweep"]
    frozen = metrics["thresholds"]["skip_below"]
    n_animal = metrics["heldout"]["all"]["n_animal"]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    fig.patch.set_facecolor("white")
    x = [r["frames_skipped_pct"] for r in sweep]
    y = [r["animals_missed"] for r in sweep]
    ax.plot(x, y, color=BLUE, linewidth=2, drawstyle="steps-post")
    last_x = -10.0
    for r in sweep:
        if round(r["skip_below"] * 100) % 10 == 0 and 0 < r["skip_below"] < 1:
            ax.plot(r["frames_skipped_pct"], r["animals_missed"], "o", color=BLUE, markersize=5, markeredgecolor="white")
            if r["frames_skipped_pct"] - last_x >= 5 and abs(r["skip_below"] - frozen) > 0.04:
                ax.annotate(f"{r['skip_below']:.1f}", (r["frames_skipped_pct"], r["animals_missed"]), textcoords="offset points",
                            xytext=(-5, 6), fontsize=8, color=INK2, ha="right")
                last_x = r["frames_skipped_pct"]
    f = min(sweep, key=lambda r: abs(r["skip_below"] - frozen))
    ax.plot(f["frames_skipped_pct"], f["animals_missed"], "o", color=ORANGE, markersize=10, markeredgecolor="white", markeredgewidth=2)
    ax.annotate(f"frozen skip_below={frozen}\n{f['frames_skipped_pct']}% of frames skipped, {f['animals_missed']} of {n_animal} animals missed",
                (f["frames_skipped_pct"], f["animals_missed"]), xytext=(4, max(y) * 0.45), textcoords="data", fontsize=9, color=INK,
                arrowprops={"arrowstyle": "-", "color": INK2, "linewidth": 0.8})
    ax.axvline(metrics["heldout"]["all"]["n_empty"] / metrics["heldout"]["all"]["n_frames"] * 100, color=GREY, linewidth=1, linestyle="--")
    ax.text(metrics["heldout"]["all"]["n_empty"] / metrics["heldout"]["all"]["n_frames"] * 100 + 0.8, max(y) * 0.95,
            "share of frames\nthat are empty", fontsize=8, color=INK2, va="top")
    ax.set_xlabel("Frames skipped, no model called (% of held-out)", color=INK2, fontsize=9)
    ax.set_ylabel(f"Animal frames skipped (of {n_animal})", color=INK2, fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=-max(1, max(y) * 0.02))
    style(ax, "The safety knob: skip more frames, miss more animals",
          f"Held-out set, {metrics['heldout']['all']['n_frames']} frames. Each dot is a skip_below value; other thresholds frozen.")
    save(fig, "tradeoff.png")


def day_night_chart(metrics: dict) -> None:
    h = metrics["heldout"]
    groups = [
        ("Empty frames skipped", "empty_skipped", "n_empty"),
        ("Animal frames missed", "animals_missed", "n_animal"),
        ("All frames skipped", "frames_skipped", "n_frames"),
    ]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    fig.patch.set_facecolor("white")
    width = 0.36
    for i, key in enumerate(("day", "night_ir")):
        s = h[key]
        xs = [g + (i - 0.5) * (width + 0.02) for g in range(len(groups))]
        vals = [100 * s[a] / s[b] if s[b] else 0 for _, a, b in groups]
        ax.bar(xs, vals, width=width, color=LIGHT[key], label=f"{LIGHT_NAME[key]} (AUC {s['presence_auc_animal_vs_empty']})")
        for x, v, (_, a, b) in zip(xs, vals, groups):
            ax.text(x, v + 1.5, f"{v:.0f}%\n{s[a]}/{s[b]}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=9, color=INK2)
    ax.set_ylim(0, 112)
    ax.set_ylabel("% of frames in the group", color=INK2, fontsize=9)
    style(ax, "Day against night (infrared), same frozen thresholds",
          "Held-out set. AUC: Decider P(animal), animal against empty frames.")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK2)
    save(fig, "day_night.png")


def draw_all(metrics: dict, frames=None) -> None:
    reliability_chart(metrics)
    tradeoff_chart(metrics)
    day_night_chart(metrics)
