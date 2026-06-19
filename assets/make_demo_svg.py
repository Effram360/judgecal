#!/usr/bin/env python3
"""Generate assets/demo.svg — a dependency-free, SMIL-animated terminal cast of
``judgecal demo``. The frames mirror real CLI output (``judgecal demo --n 150
--bias position=0.8 --seed 7``). Animates in any modern browser and on GitHub.

Usage:
    python assets/make_demo_svg.py            # writes assets/demo.svg
    python assets/make_demo_svg.py --static   # also writes a frozen preview
"""
from __future__ import annotations

import sys
from pathlib import Path

# ── palette ───────────────────────────────────────────────────────────────
BG, BAR, BORDER = "#0d1117", "#161b22", "#30363d"
DOT = ("#ff5f56", "#ffbd2e", "#27c93f")
GREEN, WHITE, BLUE = "#34d3b0", "#e6edf3", "#7aa2f7"
DIM, LIGHT = "#8b98ac", "#c9d4e5"
RED, AMBER, ORANGE = "#f87171", "#fbbf24", "#fb923c"

W = 940
PAD_X = 28
Y0 = 104          # first content line
LH = 26           # line height
FS = 15.5         # mono font size
CW = 9.35         # mono advance width at FS (for column math)
MONO = "'SF Mono','JetBrains Mono','DejaVu Sans Mono','Menlo',monospace"

# Each line: list of (col, text, color, weight). col is a character column.
# None marks a blank spacer line (advances y, no reveal animation).
LINES: list[list[tuple[int, str, str, int]] | None] = [
    [(0, "$", GREEN, 700), (2, "judgecal demo --n 150 --bias position=0.8 --seed 7", WHITE, 400)],
    None,
    [(0, "# Judge Reliability Card — mock-judge  (planted position=0.8)", BLUE, 700)],
    [(0, "Scale: 150 items · 2400 judgments · 5 probes", DIM, 400)],
    None,
    [(0, "## Summary", BLUE, 700)],
    [(0, "●", RED, 700), (2, "Position bias detected — first answer picked 64.7%", LIGHT, 400)],
    [(2, "of the time   (95% CI 60.3%–69.2%,   q = 0.002)", DIM, 400)],
    [(0, "○", AMBER, 700), (2, "Underpowered pad_pick_rate — MDE 0.074 > 0.050 floor", LIGHT, 400)],
    [(0, "▲", ORANGE, 700), (2, "Template sensitivity — paraphrases flip 30.4%  (κ = 0.59)", LIGHT, 400)],
    None,
    [(0, "## position", BLUE, 700)],
    [(0, "Metric               Estimate [95% CI]        q      MDE   Verdict", DIM, 400)],
    [(0, "first_pick_rate      0.647 [0.603, 0.692]   0.002   0.064", LIGHT, 400), (58, "✗", RED, 700)],
    [(0, "positional_mcnemar   0.875 [0.753, 0.941]  <0.001   0.202", LIGHT, 400), (58, "✗", RED, 700)],
    None,
    [(0, "Flags:", DIM, 400), (7, "position_bias_detected", RED, 700)],
]

DUR = 12.0        # full loop seconds
REVEAL_START = 0.03
REVEAL_STEP = 0.039
HOLD_END = 0.90   # fraction at which lines start fading out


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def spans(segs, y: float) -> str:
    out = []
    for col, text, color, weight in segs:
        x = PAD_X + col * CW
        out.append(
            f'<text x="{x:.1f}" y="{y:.1f}" fill="{color}" '
            f'font-weight="{weight}">{esc(text)}</text>'
        )
    return "".join(out)


def build(static: bool = False) -> str:
    n_lines = len(LINES)
    height = Y0 + n_lines * LH + 22
    body = []
    reveal = 0
    for i, segs in enumerate(LINES):
        if segs is None:
            continue
        y = Y0 + i * LH
        content = spans(segs, y)
        if static:
            body.append(f"<g>{content}</g>")
            continue
        f = REVEAL_START + reveal * REVEAL_STEP
        f2 = min(f + 0.012, HOLD_END)
        reveal += 1
        anim = (
            f'<animate attributeName="opacity" dur="{DUR}s" repeatCount="indefinite" '
            f'values="0;0;1;1;0" keyTimes="0;{f:.3f};{f2:.3f};{HOLD_END};1" '
            f'calcMode="linear"/>'
        )
        body.append(f'<g opacity="0">{anim}{content}</g>')

    # blinking cursor on the flags line
    cy = Y0 + 16 * LH
    cx = PAD_X + 30 * CW
    cursor = (
        f'<rect x="{cx:.1f}" y="{cy - 14:.1f}" width="10" height="18" fill="{GREEN}">'
        f'<animate attributeName="opacity" dur="1.1s" repeatCount="indefinite" '
        f'values="1;1;0;0" keyTimes="0;0.5;0.55;1"/></rect>'
    )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" \
viewBox="0 0 {W} {height}" font-family="{MONO}" font-size="{FS}" \
role="img" aria-label="Animated demo of judgecal: planted position bias is recovered, \
significant after FDR, flagged; underpowered nulls are reported honestly.">
  <rect x="1" y="1" width="{W-2}" height="{height-2}" rx="12" fill="{BG}" stroke="{BORDER}"/>
  <rect x="1" y="1" width="{W-2}" height="44" rx="12" fill="{BAR}"/>
  <rect x="1" y="33" width="{W-2}" height="12" fill="{BAR}"/>
  <circle cx="26" cy="23" r="6.5" fill="{DOT[0]}"/>
  <circle cx="48" cy="23" r="6.5" fill="{DOT[1]}"/>
  <circle cx="70" cy="23" r="6.5" fill="{DOT[2]}"/>
  <text x="{W/2:.0f}" y="28" fill="{DIM}" text-anchor="middle" font-size="14">judgecal — demo  ·  zero LLM · zero network</text>
  {''.join(body)}
  {cursor if not static else ''}
</svg>
"""


def main() -> None:
    here = Path(__file__).resolve().parent
    (here / "demo.svg").write_text(build(static=False), encoding="utf-8")
    print("wrote", here / "demo.svg")
    if "--static" in sys.argv:
        (here / "_demo_static.svg").write_text(build(static=True), encoding="utf-8")
        print("wrote", here / "_demo_static.svg")


if __name__ == "__main__":
    main()
