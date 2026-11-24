# ruff: noqa: E501 - SVG markup is built from long f-strings; splitting them hurts readability
"""Dependency-free SVG charts for the experiment report (no plotting library is installed).

Only what the report needs: grouped bars, a forest plot of paired differences with intervals, and
a chunk-size range plot. Colours are fixed, colour-blind-safe and always paired with a text label.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape

COLORS = {"A": "#1b6ca8", "B": "#d95f02", "C": "#7570b3", "D": "#1b9e77"}
INK = "#222222"
GRID = "#d0d0d0"


def _open(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="sans-serif" font-size="12">',
        f"<title>{escape(title)}</title>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="18" text-anchor="middle" font-size="14" fill="{INK}">'
        f"{escape(title)}</text>",
    ]


def grouped_bars(
    title: str,
    categories: Sequence[str],
    series: Mapping[str, Sequence[float]],
    *,
    y_label: str,
    y_max: float,
) -> str:
    """Bars per category (x) with one bar per series (arm). Values are annotated."""
    width, height = 640, 360
    left, right, top, bottom = 56, 16, 36, 56
    plot_w, plot_h = width - left - right, height - top - bottom
    out = _open(width, height, title)
    for tick in range(0, 6):
        value = y_max * tick / 5
        y = top + plot_h * (1 - tick / 5)
        out.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="{GRID}"/>'
        )
        out.append(
            f'<text x="{left - 6}" y="{y + 4:.1f}" text-anchor="end" fill="{INK}">{value:.2g}</text>'
        )
    names = list(series)
    group_w = plot_w / len(categories)
    bar_w = group_w * 0.8 / len(names)
    for i, category in enumerate(categories):
        x0 = left + i * group_w + group_w * 0.1
        for j, name in enumerate(names):
            value = series[name][i]
            h = plot_h * min(value / y_max, 1.0)
            x = x0 + j * bar_w
            out.append(
                f'<rect x="{x:.1f}" y="{top + plot_h - h:.1f}" width="{bar_w - 2:.1f}" '
                f'height="{h:.1f}" fill="{COLORS.get(name, "#888")}"><title>{escape(name)} '
                f"{escape(category)}: {value:.3g}</title></rect>"
            )
            out.append(
                f'<text x="{x + bar_w / 2 - 1:.1f}" y="{top + plot_h - h - 3:.1f}" '
                f'text-anchor="middle" font-size="10" fill="{INK}">{value:.3g}</text>'
            )
        out.append(
            f'<text x="{left + (i + 0.5) * group_w:.1f}" y="{height - bottom + 16}" '
            f'text-anchor="middle" fill="{INK}">{escape(category)}</text>'
        )
    for j, name in enumerate(names):
        lx = left + j * 70
        out.append(
            f'<rect x="{lx}" y="{height - 22}" width="12" height="12" fill="{COLORS.get(name, "#888")}"/>'
        )
        out.append(f'<text x="{lx + 16}" y="{height - 12}" fill="{INK}">Arm {escape(name)}</text>')
    out.append(
        f'<text transform="translate(14 {top + plot_h / 2}) rotate(-90)" text-anchor="middle" '
        f'fill="{INK}">{escape(y_label)}</text>'
    )
    out.append("</svg>")
    return "\n".join(out)


def forest(title: str, rows: Sequence[tuple[str, float, float, float]], *, x_label: str) -> str:
    """Point estimates with intervals: rows of ``(label, estimate, lo, hi)`` around a zero line."""
    width = 640
    row_h = 26
    left, right, top = 170, 24, 40
    height = top + row_h * len(rows) + 48
    lo_all = min([0.0] + [r[2] for r in rows])
    hi_all = max([0.0] + [r[3] for r in rows])
    span = (hi_all - lo_all) or 1.0
    lo_all -= span * 0.08
    hi_all += span * 0.08
    plot_w = width - left - right

    def x_of(v: float) -> float:
        return left + plot_w * (v - lo_all) / (hi_all - lo_all)

    out = _open(width, height, title)
    out.append(
        f'<line x1="{x_of(0):.1f}" y1="{top - 6}" x2="{x_of(0):.1f}" y2="{top + row_h * len(rows)}" '
        f'stroke="{INK}" stroke-dasharray="4 3"/>'
    )
    for i, (label, est, lo, hi) in enumerate(rows):
        y = top + row_h * i + row_h / 2
        out.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" fill="{INK}">{escape(label)}</text>'
        )
        out.append(
            f'<line x1="{x_of(lo):.1f}" y1="{y:.1f}" x2="{x_of(hi):.1f}" y2="{y:.1f}" stroke="{INK}" stroke-width="2"/>'
        )
        out.append(
            f'<circle cx="{x_of(est):.1f}" cy="{y:.1f}" r="4.5" fill="{COLORS["B"]}">'
            f"<title>{escape(label)}: {est:+.2f} [{lo:+.2f}, {hi:+.2f}]</title></circle>"
        )
        out.append(
            f'<text x="{width - right}" y="{y + 4:.1f}" text-anchor="end" font-size="10" fill="{INK}">'
            f"{est:+.1f} [{lo:+.1f}, {hi:+.1f}]</text>"
        )
    out.append(
        f'<text x="{left + plot_w / 2}" y="{height - 10}" text-anchor="middle" fill="{INK}">{escape(x_label)}</text>'
    )
    out.append("</svg>")
    return "\n".join(out)


def range_plot(
    title: str,
    rows: Sequence[tuple[str, float, float, float, float, float]],
    *,
    x_label: str,
) -> str:
    """Per-arm distribution summary: ``(label, min, p25, median, p75, max)`` as whisker + box."""
    width, row_h, left, right, top = 640, 40, 110, 20, 40
    height = top + row_h * len(rows) + 48
    x_max = max(r[5] for r in rows) * 1.05
    plot_w = width - left - right

    def x_of(v: float) -> float:
        return left + plot_w * v / x_max

    out = _open(width, height, title)
    for tick in range(0, 6):
        v = x_max * tick / 5
        out.append(
            f'<line x1="{x_of(v):.1f}" y1="{top - 4}" x2="{x_of(v):.1f}" y2="{top + row_h * len(rows)}" stroke="{GRID}"/>'
        )
        out.append(
            f'<text x="{x_of(v):.1f}" y="{top + row_h * len(rows) + 14}" text-anchor="middle" fill="{INK}">{v:.0f}</text>'
        )
    for i, (label, mn, p25, med, p75, mx) in enumerate(rows):
        y = top + row_h * i + row_h / 2
        color = COLORS.get(label.split()[0], "#888")
        out.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" fill="{INK}">{escape(label)}</text>'
        )
        out.append(
            f'<line x1="{x_of(mn):.1f}" y1="{y:.1f}" x2="{x_of(mx):.1f}" y2="{y:.1f}" stroke="{INK}"/>'
        )
        out.append(
            f'<rect x="{x_of(p25):.1f}" y="{y - 9:.1f}" width="{max(x_of(p75) - x_of(p25), 1):.1f}" '
            f'height="18" fill="{color}" fill-opacity="0.55" stroke="{color}"><title>{escape(label)}: '
            f"min {mn:.0f}, p25 {p25:.0f}, median {med:.0f}, p75 {p75:.0f}, max {mx:.0f}</title></rect>"
        )
        out.append(
            f'<line x1="{x_of(med):.1f}" y1="{y - 9:.1f}" x2="{x_of(med):.1f}" y2="{y + 9:.1f}" stroke="{INK}" stroke-width="2"/>'
        )
    out.append(
        f'<text x="{left + plot_w / 2}" y="{height - 8}" text-anchor="middle" fill="{INK}">{escape(x_label)}</text>'
    )
    out.append("</svg>")
    return "\n".join(out)
