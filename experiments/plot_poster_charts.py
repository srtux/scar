"""Generate the 4 print-ready 300-DPI poster charts for Emotion-Shift Memory (`results/esm_results.json`)."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_PATH = os.path.join(REPO_ROOT, "results", "esm_results.json")
CHARTS_DIR = os.path.join(REPO_ROOT, "results", "charts")

# Clean, accessible scientific poster palette (Radix/Primer-aligned Slate, Blue, Jade, Amber, Red)
NAVY = "#1E3A8A"
BLUE = "#2563EB"
JADE = "#059669"
AMBER = "#D97706"
RED = "#DC2626"
SLATE_DARK = "#0F172A"
SLATE_MID = "#475569"
SLATE_LIGHT = "#CBD5E1"
BG_GRID = "#E2E8F0"


def _apply_poster_rc() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "axes.edgecolor": "#94A3B8",
            "axes.linewidth": 1.1,
            "axes.titlesize": 13.2,
            "axes.titleweight": "bold",
            "axes.labelsize": 11.8,
            "axes.labelweight": "bold",
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _style_axes(ax: Any, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, linestyle="--", linewidth=0.8, color=BG_GRID, alpha=0.85, zorder=0)
    ax.set_axisbelow(True)


def plot_chart1_silent_failures(data: Dict[str, Any], out_dir: str) -> str:
    """Chart 1 (Left Panel): Silent Failures vs. Tool Crashes across tau2-bench."""
    sf = data["chart1_silent_failures"]
    slices = [
        ("Airline\n(n = 800 runs)", sf["airline"]),
        ("Retail\n(n = 1,824 runs)", sf["retail"]),
        ("Combined\n(n = 2,624 runs)", sf["combined"]),
    ]

    labels = [s[0] for s in slices]
    silent_pcts = [s[1]["silent_failure_rate"] * 100.0 for s in slices]
    crash_pcts = [100.0 - p for p in silent_pcts]
    fail_counts = [s[1]["failed_runs"] for s in slices]
    silent_counts = [s[1]["silent_failures"] for s in slices]

    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=300)
    _style_axes(ax, grid_axis="y")

    x = range(len(labels))
    width = 0.54
    b1 = ax.bar(
        x,
        silent_pcts,
        width=width,
        color=BLUE,
        edgecolor="white",
        linewidth=1.2,
        label="Silent Failures (0 Tool Crashes)",
        zorder=3,
    )
    ax.bar(
        x,
        crash_pcts,
        width=width,
        bottom=silent_pcts,
        color=SLATE_LIGHT,
        edgecolor="white",
        linewidth=1.2,
        label="Explicit Tool Crashes (>=1 Error)",
        zorder=3,
    )

    for idx, rect in enumerate(b1):
        sp = silent_pcts[idx]
        cp = crash_pcts[idx]
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            sp / 2.0,
            f"{sp:.1f}%\n({silent_counts[idx]}/{fail_counts[idx]})",
            ha="center",
            va="center",
            color="white",
            fontsize=11.5,
            fontweight="bold",
            zorder=4,
        )
        if cp >= 10.0:
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                sp + cp / 2.0,
                f"{cp:.1f}%",
                ha="center",
                va="center",
                color=SLATE_DARK,
                fontsize=10.5,
                fontweight="bold",
                zorder=4,
            )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontweight="bold", color=SLATE_DARK)
    ax.set_ylim(0, 122)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.set_ylabel("Share of Failed Agent Conversations", color=SLATE_DARK)
    ax.set_title(
        "Most Agent Failures Leave Zero Software Crash Logs",
        pad=26,
        color=SLATE_DARK,
    )
    ax.text(
        0.5,
        1.02,
        "Share of failed tau2-bench runs across 4 LLMs with 0 vs. >=1 tool errors",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10.2,
        color=SLATE_MID,
    )
    ax.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#CBD5E1",
        fontsize=9.8,
    )

    fig.tight_layout()
    png_path = os.path.join(out_dir, "chart1_silent_failures.png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.16)
    fig.savefig(os.path.join(out_dir, "chart1_silent_failures.pdf"), bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)
    return png_path


def plot_chart2_detector_pareto(data: Dict[str, Any], out_dir: str) -> str:
    """Chart 2 (Center Panel): Per-Turn Latency vs. Human Emotion-Shift Detection F1."""
    dets = data["stage1_shift_detection"]["wildfeedback"]["detectors"]
    order = [
        ("VADER", "VADER\n(Word Lexicon)", "#64748B", "s", (14, -18), "left"),
        ("GoEmotions", "GoEmotions\n(28-Emotion RoBERTa)", AMBER, "^", (-10, 32), "center"),
        ("Laya", "Laya\n(Local ModernBERT)", JADE, "D", (-14, 14), "right"),
        ("Flash-Lite", "Gemini 3.5 Flash-Lite\n(Cloud LLM)", BLUE, "o", (0, -34), "center"),
    ]

    fig, ax = plt.subplots(figsize=(8.2, 5.1), dpi=300)
    _style_axes(ax, grid_axis="both")
    ax.set_xscale("log")

    xs: List[float] = []
    ys: List[float] = []
    for key, _, _, _, _, _ in order:
        d = dets[key]
        xs.append(max(0.02, float(d["ms_per_turn"])))
        ys.append(float(d["fixed_f1"]) * 100.0)

    # Dashed progression curve connecting the 4 detectors sorted by latency
    pts_sorted = sorted(zip(xs, ys), key=lambda p: p[0])
    ax.plot(
        [p[0] for p in pts_sorted],
        [p[1] for p in pts_sorted],
        linestyle="--",
        linewidth=1.6,
        color="#94A3B8",
        zorder=2,
    )

    for (key, display, color, marker, offset, ha), x_val, y_val in zip(order, xs, ys):
        ax.scatter(
            [x_val],
            [y_val],
            s=175,
            color=color,
            marker=marker,
            edgecolors="white",
            linewidths=1.6,
            zorder=4,
        )
        lat_str = f"{x_val:.2f} ms" if x_val < 1.0 else f"{x_val:.0f} ms"
        label_txt = f"{display}\nF1: {y_val:.1f}% ({lat_str}/turn)"
        ax.annotate(
            label_txt,
            xy=(x_val, y_val),
            xytext=offset,
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color=SLATE_DARK,
            bbox={
                "boxstyle": "round,pad=0.32",
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 1.2,
                "alpha": 0.96,
            },
            zorder=5,
        )

    min_x = min(xs) * 0.32
    max_x = max(xs) * 3.8
    ax.set_xlim(min_x, max_x)
    min_y = max(0.0, min(ys) - 14.0)
    max_y = min(100.0, max(ys) + 14.0)
    ax.set_ylim(min_y, max_y)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.set_xlabel("Latency per User Turn (ms, log scale - left is faster)", color=SLATE_DARK)
    ax.set_ylabel("Human 'Frustrated -> Fixed' F1", color=SLATE_DARK)
    n_threads = data["stage1_shift_detection"]["wildfeedback"]["total_threads"]
    ax.set_title(
        "Context-Aware Shift Gates Catch Calm Fixes Dictionaries Miss",
        pad=26,
        color=SLATE_DARK,
    )
    ax.text(
        0.5,
        1.02,
        f"Evaluated on {n_threads} real human-annotated multi-turn WildFeedback chats",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10.2,
        color=SLATE_MID,
    )

    fig.tight_layout()
    png_path = os.path.join(out_dir, "chart2_detector_pareto.png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.16)
    fig.savefig(os.path.join(out_dir, "chart2_detector_pareto.pdf"), bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)
    return png_path


def plot_chart3_memory_ablation(data: Dict[str, Any], out_dir: str) -> str:
    """Chart 3 (Right Panel Top): 5-Condition MEMORY.md Ablation on Held-Out Unseen Tasks."""
    conds = data["stage2_mistake_learning"]["conditions"]
    spec = [
        ("no_memory", "No Memory\n(Zero-Shot)", "#94A3B8"),
        ("routine_chats", "Routine\n(Calm -> Sat)", "#64748B"),
        ("negative_only", "Negative-Only\n(Unresolved)", RED),
        ("all_chats", "All Chats\n(Unfiltered)", AMBER),
        ("emotion_shift", "Emotion-Shift\n(Fixed Only)", JADE),
    ]

    labels = [s[1] for s in spec]
    rates = [float(conds[s[0]]["repeat_error_rate"]) * 100.0 for s in spec]
    counts = [
        (int(conds[s[0]]["mistakes_repeated"]), int(conds[s[0]]["total_heldout_trials"]))
        for s in spec
    ]
    colors = [s[2] for s in spec]

    fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=300)
    _style_axes(ax, grid_axis="y")

    x = range(len(spec))
    bars = ax.bar(
        x,
        rates,
        width=0.56,
        color=colors,
        edgecolor="white",
        linewidth=1.2,
        zorder=3,
    )

    # Highlight baseline dashed reference line
    ax.axhline(rates[0], linestyle=":", linewidth=1.3, color="#475569", zorder=2)

    for idx, rect in enumerate(bars):
        r_val = rates[idx]
        rep, tot = counts[idx]
        weight = "bold"
        txt_color = JADE if spec[idx][0] == "emotion_shift" else SLATE_DARK
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            r_val + 1.4,
            f"{r_val:.1f}%\n({rep}/{tot})",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight=weight,
            color=txt_color,
            bbox={
                "boxstyle": "round,pad=0.14",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.9,
            },
            zorder=4,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold", color=SLATE_DARK)
    ax.set_ylim(0, max(rates) + 14.0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.set_ylabel("Held-Out Mistake Rate (Lower is Better)", color=SLATE_DARK)
    ax.set_title(
        "What Should Enter MEMORY.md? Fixed Mistakes Beat Raw Volume",
        pad=26,
        color=SLATE_DARK,
    )
    budget = data["stage2_mistake_learning"]["memory_budget_tokens"]
    ctrl = data["stage2_mistake_learning"].get("non_regression_controls", {})
    esm_ctrl = ctrl.get("emotion_shift", {}).get("pass_rate")
    zs_ctrl = ctrl.get("no_memory", {}).get("pass_rate")
    sub = f"2-fold unseen tau2-bench tasks (<= {budget}-token MEMORY.md)"
    if esm_ctrl is not None and zs_ctrl is not None:
        sub += f" | Routine control pass: {esm_ctrl*100:.0f}% vs {zs_ctrl*100:.0f}%"
    ax.text(
        0.5,
        1.02,
        sub,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.8,
        color=SLATE_MID,
    )

    fig.tight_layout()
    png_path = os.path.join(out_dir, "chart3_memory_ablation.png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.16)
    fig.savefig(os.path.join(out_dir, "chart3_memory_ablation.pdf"), bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)
    return png_path


def plot_chart4_learning_curve(data: Dict[str, Any], out_dir: str) -> str:
    """Chart 4 (Right Panel Bottom): Mistake Rate vs. Past Training Conversations Observed."""
    lc = data["stage2_mistake_learning"]["learning_curve"]
    steps: List[int] = [int(x) for x in lc["chat_counts"]]

    series_spec = [
        ("negative_only", "Negative-Only (Unresolved)", RED, "^", "--"),
        ("all_chats", "All Chats (Unfiltered)", AMBER, "s", "-."),
        ("emotion_shift", "Emotion-Shift Memory (Ours)", JADE, "o", "-"),
    ]

    fig, ax = plt.subplots(figsize=(8.4, 4.9), dpi=300)
    _style_axes(ax, grid_axis="both")

    all_vals: List[float] = []
    endpoints: List[Tuple[float, str, str]] = []
    for key, label, color, marker, ls in series_spec:
        ys = [float(v) * 100.0 for v in lc["series"][key]]
        all_vals.extend(ys)
        lw = 3.0 if key == "emotion_shift" else 2.2
        ax.plot(
            steps,
            ys,
            label=label,
            color=color,
            marker=marker,
            markersize=8.5,
            linestyle=ls,
            linewidth=lw,
            zorder=4 if key == "emotion_shift" else 3,
        )
        endpoints.append((ys[-1], f"{ys[-1]:.1f}%", color))

    # Deterministic vertical de-collision for right-edge endpoint labels
    endpoints_sorted = sorted(enumerate(endpoints), key=lambda t: t[1][0])
    dy_offsets = [0.0] * len(endpoints)
    for rank in range(1, len(endpoints_sorted)):
        prev_idx, (prev_y, _, _) = endpoints_sorted[rank - 1]
        cur_idx, (cur_y, _, _) = endpoints_sorted[rank]
        if (cur_y + dy_offsets[cur_idx]) - (prev_y + dy_offsets[prev_idx]) < 2.2:
            dy_offsets[prev_idx] -= 5.0
            dy_offsets[cur_idx] += 5.0

    for idx, (y_end, txt, color) in enumerate(endpoints):
        ax.annotate(
            txt,
            xy=(steps[-1], y_end),
            xytext=(8, dy_offsets[idx]),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=10.2,
            fontweight="bold",
            color=color,
        )

    ax.set_xticks(steps)
    ax.set_xlim(-4, steps[-1] + 17)
    lo = max(0.0, min(all_vals) - 6.0)
    hi = min(100.0, max(all_vals) + 10.0)
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.set_xlabel("Past Training Conversations Observed per Fold", color=SLATE_DARK)
    ax.set_ylabel("Held-Out Mistake Rate (Lower is Better)", color=SLATE_DARK)
    ax.set_title(
        "Mistake Rate Falls as MEMORY.md Accumulates Fixed Mistakes",
        pad=26,
        color=SLATE_DARK,
    )
    ax.text(
        0.5,
        1.02,
        "Held-out error rate across 0, 25, 50, and 100 observed past chats (400-token cap)",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10.2,
        color=SLATE_MID,
    )
    ax.legend(
        loc="best",
        frameon=True,
        facecolor="white",
        edgecolor="#CBD5E1",
        fontsize=10.0,
    )

    fig.tight_layout()
    png_path = os.path.join(out_dir, "chart4_learning_curve.png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.16)
    fig.savefig(os.path.join(out_dir, "chart4_learning_curve.pdf"), bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)
    return png_path


def generate_all_charts(results_path: str = RESULTS_PATH, out_dir: str = CHARTS_DIR) -> List[str]:
    _apply_poster_rc()
    os.makedirs(out_dir, exist_ok=True)
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    paths = [
        plot_chart1_silent_failures(data, out_dir),
        plot_chart2_detector_pareto(data, out_dir),
        plot_chart3_memory_ablation(data, out_dir),
        plot_chart4_learning_curve(data, out_dir),
    ]
    for p in paths:
        print(f"Saved chart: {p}")
    return paths


if __name__ == "__main__":
    generate_all_charts()
