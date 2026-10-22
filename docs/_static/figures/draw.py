#!/usr/bin/env python3
"""Emit Excalidraw JSON, SVG, and GIF files for the documentation."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
README_ASSETS = HERE.parents[2] / "assets"

BG = "#FFF6DC"
INK = "#2B2118"
CORAL = "#FF6B6B"
TEAL = "#2EC4B6"
SUN = "#FFD166"
PURPLE = "#7B6CF6"
PINK = "#FF8FAB"
NAVY = "#1B4B6B"
GREEN = "#6BCB77"
ORANGE = "#FF9F43"
CREAM = "#FFF1C2"
WHITE = "#FFFEF8"
SKY = "#8EC5FF"


def _id() -> str:
    return f"fe{random.randrange(16**8):08x}"


class Scene:
    def __init__(self, name: str, w: int, h: int, title: str) -> None:
        self.name, self.w, self.h, self.title = name, w, h, title
        self.svg: list[str] = []
        self.elements: list[dict] = []
        random.seed(sum(ord(c) for c in name) * 997)

    def _rot(self, amp: float = 0.25) -> float:
        return random.uniform(-amp, amp)

    def _ex(self, kind: str, x: float, y: float, w: float, h: float, **kw) -> None:
        el = {
            "id": _id(),
            "type": kind,
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "angle": 0,
            "strokeColor": kw.get("stroke", INK),
            "backgroundColor": kw.get("fill", "transparent"),
            "fillStyle": "solid",
            "strokeWidth": 2,
            "roughness": 1,
            "opacity": 100,
            "seed": random.randrange(1, 99_999),
            "version": 1,
            "versionNonce": random.randrange(1, 99_999),
            "isDeleted": False,
            "boundElements": None,
            "updated": 1,
            "link": None,
            "locked": False,
        }
        text = kw.pop("text", "")
        el.update({k: v for k, v in kw.items() if k not in ("fill", "stroke")})
        self.elements.append(el)
        if text:
            self.elements.append(
                {
                    **el,
                    "id": _id(),
                    "type": "text",
                    "x": x + 6,
                    "y": y + max(4, h / 2 - 10),
                    "width": w - 12,
                    "height": 22,
                    "text": text.replace("|", " "),
                    "originalText": text.replace("|", " "),
                    "fontSize": 16,
                    "fontFamily": 2,
                    "textAlign": "center",
                    "verticalAlign": "middle",
                    "backgroundColor": "transparent",
                    "strokeColor": INK,
                    "containerId": None,
                    "lineHeight": 1.25,
                }
            )

    def star(self, x: float, y: float, r: float = 9, fill: str = SUN) -> None:
        pts = []
        for i in range(10):
            ang = math.radians(-90 + i * 36)
            rad = r if i % 2 == 0 else r * 0.42
            pts.append(f"{x + rad * math.cos(ang):.1f},{y + rad * math.sin(ang):.1f}")
        self.svg.append(
            f'<polygon points="{" ".join(pts)}" fill="{fill}" stroke="{INK}" stroke-width="1.6"/>'
        )

    def stamp(self, x: float, y: float, text: str, fill: str = CORAL) -> None:
        rot = random.uniform(-12, -6)
        w = 9 * len(text) + 22
        self.svg.append(
            f'<g transform="rotate({rot:.1f} {x:.0f} {y:.0f})">'
            f'<ellipse cx="{x:.0f}" cy="{y:.0f}" rx="{w / 2:.0f}" ry="18" fill="{fill}" '
            f'stroke="{INK}" stroke-width="2.4"/>'
            f'<text x="{x:.0f}" y="{y + 5:.0f}" text-anchor="middle" font-size="13" '
            f'font-family="Trebuchet MS, Segoe UI, sans-serif" font-weight="800" '
            f'fill="{WHITE}">{text}</text></g>'
        )
        self._ex("ellipse", x - w / 2, y - 18, w, 36, fill=fill, stroke=INK, text=text)

    def doodle_bot(self, x: float, y: float) -> None:
        self.svg.append(
            f'<g stroke="{INK}" stroke-width="2.2" fill="{SKY}">'
            f'<rect x="{x}" y="{y}" width="36" height="28" rx="6"/>'
            f'<circle cx="{x + 12}" cy="{y + 12}" r="3.2" fill="{INK}"/>'
            f'<circle cx="{x + 24}" cy="{y + 12}" r="3.2" fill="{INK}"/>'
            f'<rect x="{x + 8}" y="{y + 28}" width="8" height="10" rx="2" fill="{TEAL}"/>'
            f'<rect x="{x + 20}" y="{y + 28}" width="8" height="10" rx="2" fill="{TEAL}"/>'
            f'<line x1="{x + 18}" y1="{y}" x2="{x + 18}" y2="{y - 10}"/>'
            f'<circle cx="{x + 18}" cy="{y - 12}" r="3.4" fill="{CORAL}"/>'
            f"</g>"
        )

    def doodle_chip(self, x: float, y: float) -> None:
        self.svg.append(
            f'<g stroke="{INK}" stroke-width="2" fill="{CREAM}">'
            f'<rect x="{x}" y="{y}" width="34" height="28" rx="4"/>'
            f'<rect x="{x + 8}" y="{y + 7}" width="18" height="14" rx="2" fill="{TEAL}"/>'
            f'<line x1="{x}" y1="{y + 8}" x2="{x - 8}" y2="{y + 8}"/>'
            f'<line x1="{x}" y1="{y + 20}" x2="{x - 8}" y2="{y + 20}"/>'
            f'<line x1="{x + 34}" y1="{y + 8}" x2="{x + 42}" y2="{y + 8}"/>'
            f'<line x1="{x + 34}" y1="{y + 20}" x2="{x + 42}" y2="{y + 20}"/>'
            f"</g>"
        )

    def doodle_cam(self, x: float, y: float) -> None:
        self.svg.append(
            f'<g stroke="{INK}" stroke-width="2" fill="{PINK}">'
            f'<rect x="{x}" y="{y + 6}" width="40" height="24" rx="6"/>'
            f'<circle cx="{x + 20}" cy="{y + 18}" r="8" fill="{WHITE}"/>'
            f'<circle cx="{x + 20}" cy="{y + 18}" r="4" fill="{NAVY}"/>'
            f'<rect x="{x + 28}" y="{y}" width="12" height="8" rx="2" fill="{SUN}"/>'
            f"</g>"
        )

    def doodle_clock(self, x: float, y: float, r: float = 22) -> None:
        self.svg.append(
            f'<g stroke="{INK}" stroke-width="2.2" fill="{SUN}">'
            f'<circle cx="{x}" cy="{y}" r="{r}"/>'
            f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y - r + 7}"/>'
            f'<line x1="{x}" y1="{y}" x2="{x + r - 9}" y2="{y + 4}"/>'
            f"</g>"
        )

    def caption(self, x: float, y: float, text: str, *, size: int = 15, anchor: str = "start", fill: str = NAVY, bold: bool = True) -> None:
        wt = 800 if bold else 600
        self.svg.append(
            f'<text x="{x:.0f}" y="{y:.0f}" text-anchor="{anchor}" font-size="{size}" '
            f'font-family="Trebuchet MS, Segoe UI, sans-serif" font-weight="{wt}" fill="{fill}">'
            f"{text}</text>"
        )

    def lines(
        self,
        x: float,
        y: float,
        texts: list[str],
        *,
        size: int = 14,
        leading: int = 20,
        fill: str = INK,
        anchor: str = "middle",
        bold: bool = False,
    ) -> None:
        for i, t in enumerate(texts):
            self.caption(x, y + i * leading, t, size=size, anchor=anchor, fill=fill, bold=bold)

    def box(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str,
        label: str,
        *,
        size: int = 18,
        stroke: str = INK,
    ) -> None:
        rot = self._rot()
        cx, cy = x + w / 2, y + h / 2
        lines = [ln for ln in label.split("|") if ln]
        self.svg.append(
            f'<g transform="rotate({rot:.2f} {cx:.0f} {cy:.0f})">'
            f'<rect x="{x + 3:.0f}" y="{y + 5:.0f}" width="{w:.0f}" height="{h:.0f}" rx="16" '
            f'fill="#E8C98A" opacity="0.45"/>'
            f'<rect x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{h:.0f}" rx="16" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="2.6"/>'
        )
        if len(lines) == 1:
            self.svg.append(
                f'<text x="{cx:.0f}" y="{cy + 6:.0f}" text-anchor="middle" font-size="{size}" '
                f'font-family="Trebuchet MS, Segoe UI, sans-serif" font-weight="800" fill="{INK}">'
                f"{lines[0]}</text>"
            )
        else:
            y0 = cy - 10 * (len(lines) - 1)
            for i, ln in enumerate(lines):
                fs = size if i == 0 else max(14, size - 3)
                wt = 800 if i == 0 else 600
                self.svg.append(
                    f'<text x="{cx:.0f}" y="{y0 + i * 22:.0f}" text-anchor="middle" font-size="{fs}" '
                    f'font-family="Trebuchet MS, Segoe UI, sans-serif" font-weight="{wt}" fill="{INK}">'
                    f"{ln}</text>"
                )
        self.svg.append("</g>")
        self._ex("rectangle", x, y, w, h, fill=fill, stroke=stroke, text=label)

    def oval(self, x: float, y: float, w: float, h: float, fill: str, label: str, size: int = 16) -> None:
        rot = self._rot(0.6)
        cx, cy = x + w / 2, y + h / 2
        self.svg.append(
            f'<g transform="rotate({rot:.2f} {cx:.0f} {cy:.0f})">'
            f'<ellipse cx="{cx:.0f}" cy="{cy:.0f}" rx="{w / 2:.0f}" ry="{h / 2:.0f}" '
            f'fill="{fill}" stroke="{INK}" stroke-width="2.6"/>'
            f'<text x="{cx:.0f}" y="{cy + 5:.0f}" text-anchor="middle" font-size="{size}" '
            f'font-family="Trebuchet MS, Segoe UI, sans-serif" font-weight="800" fill="{INK}">'
            f"{label}</text></g>"
        )
        self._ex("ellipse", x, y, w, h, fill=fill, stroke=INK, text=label)

    def note(self, x: float, y: float, text: str, *, fill: str = SUN, size: int = 14) -> None:
        w = 8.6 * len(text) + 28
        h = 36
        rot = random.uniform(-1.6, 1.2)
        self.svg.append(
            f'<g transform="rotate({rot:.1f} {x:.0f} {y:.0f})">'
            f'<rect x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{h:.0f}" rx="9" '
            f'fill="{fill}" stroke="{INK}" stroke-width="2.2"/>'
            f'<text x="{x + w / 2:.0f}" y="{y + 24:.0f}" text-anchor="middle" font-size="{size}" '
            f'font-family="Trebuchet MS, Segoe UI, sans-serif" font-weight="800" fill="{INK}">'
            f"{text}</text></g>"
        )
        self._ex("rectangle", x, y, w, h, fill=fill, stroke=INK, text=text)

    def arrow(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        color: str = INK,
        dashed: bool = False,
    ) -> None:
        dash = ' stroke-dasharray="8 6"' if dashed else ""
        mx = (x1 + x2) / 2 + random.uniform(-10, 10)
        my = (y1 + y2) / 2 + random.uniform(-8, 8)
        self.svg.append(
            f'<path d="M {x1:.0f} {y1:.0f} Q {mx:.0f} {my:.0f} {x2:.0f} {y2:.0f}" '
            f'fill="none" stroke="{color}" stroke-width="2.8"{dash} '
            f'marker-end="url(#{self.name}-arr)"/>'
        )
        self._ex(
            "arrow",
            min(x1, x2),
            min(y1, y2),
            abs(x2 - x1) or 10,
            abs(y2 - y1) or 10,
            fill="transparent",
            stroke=color,
        )

    def curve(self, d: str, *, color: str = CORAL, width: float = 4.0) -> None:
        self.svg.append(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" '
            f'stroke-linecap="round"/>'
        )

    def write(self) -> None:
        svg = [
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
                f'width="{self.w}" height="{self.h}" role="img">'
            ),
            f"<title>{self.title}</title>",
            "<defs>",
            (
                f"<marker id='{self.name}-arr' markerWidth='10' markerHeight='10' refX='8' refY='3.2' "
                "orient='auto'><path d='M0,0 L10,3.2 L0,6.4 Z' fill='#2B2118'/></marker>"
            ),
            "</defs>",
            f'<rect width="100%" height="100%" rx="22" fill="{BG}" stroke="#E8D7A8" stroke-width="3"/>',
            (
                f'<text x="28" y="38" font-size="22" font-family="Trebuchet MS, Segoe UI, sans-serif" '
                f'font-weight="800" fill="{NAVY}">{self.title}</text>'
            ),
        ]
        svg.extend(self.svg)
        svg.append("</svg>")
        (HERE / f"{self.name}.svg").write_text("\n".join(svg), encoding="utf-8")
        (HERE / f"{self.name}.excalidraw").write_text(
            json.dumps(
                {
                    "type": "excalidraw",
                    "version": 2,
                    "source": "https://excalidraw.com",
                    "elements": self.elements,
                    "appState": {"viewBackgroundColor": BG, "gridSize": None},
                    "files": {},
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def purpose() -> None:
    s = Scene("purpose", 1180, 400, "who owns what")
    s.caption(590, 58, "one robot loop  ·  three owners  ·  not a pipeline", size=16, anchor="middle")

    s.box(36, 78, 350, 220, SUN, "")
    s.doodle_chip(191, 96)
    s.caption(211, 156, "Hardware", size=24, anchor="middle")
    s.lines(211, 188, ["owns the period", "CPU / Jetson / SO-100", "motors on the wire"], size=16)
    s.caption(211, 266, "not e-stop, not cameras", size=14, anchor="middle", fill=NAVY, bold=False)

    s.box(414, 78, 350, 220, TEAL, "")
    s.doodle_bot(571, 94)
    s.caption(589, 156, "FlowEdge", size=24, anchor="middle")
    s.lines(589, 188, ["Mamba + flow ODE", "or DP U-Net", "fixed RSS, no malloc"], size=16)
    s.caption(589, 266, "not training, not safety", size=14, anchor="middle", fill=NAVY, bold=False)

    s.box(792, 78, 350, 220, PINK, "")
    s.doodle_cam(947, 96)
    s.caption(967, 156, "LeRobot", size=24, anchor="middle")
    s.lines(967, 188, ["train the policy", "cameras + encoder", "driver, limits, e-stop"], size=16)
    s.caption(967, 266, "not native kernels", size=14, anchor="middle", fill=NAVY, bold=False)

    s.note(70, 312, "the 10 ms tick", fill=SKY)
    s.note(470, 312, "replayable action", fill=ORANGE)
    s.note(848, 312, "plugin calls Core", fill=CORAL)
    s.caption(590, 378, "LeRobot observes  →  FlowEdge infers  →  Hardware actuates", size=15, anchor="middle")
    s.write()


def flow_matching() -> None:
    s = Scene("flow-matching", 960, 380, "flow matching    dx/dt = v(x, t | c)")
    s.oval(36, 140, 168, 100, SUN, "x0  noise")
    s.oval(750, 130, 180, 120, GREEN, "x1  action")
    s.curve("M 204 190 C 340 40, 560 330, 750 190", color=CORAL, width=5)
    s.arrow(310, 95, 380, 78, color=PURPLE)
    s.arrow(490, 230, 560, 205, color=PURPLE)
    s.arrow(640, 145, 720, 165, color=PURPLE)
    s.caption(360, 68, "velocity arrows  ·  Euler / Heun / RK4")
    s.box(280, 280, 400, 64, CREAM, "condition  c   from Mamba (or your encoder)")
    s.note(40, 48, "train on a straight path", fill=SUN)
    s.note(780, 48, "t :  0  →  1", fill=PINK)
    s.star(470, 150, 10, SUN)
    s.write()


def workflow() -> None:
    s = Scene("workflow", 980, 270, "everyday loop")
    boxes = [
        (24, 78, 155, 100, SUN, "convert|.safetensors"),
        (210, 78, 165, 100, TEAL, "load arena|fixed RSS"),
        (406, 78, 185, 100, PURPLE, "sample head|flow or DDIM"),
        (622, 78, 155, 100, ORANGE, "period?|hold / drop"),
        (808, 78, 148, 100, GREEN, "act|motors"),
    ]
    for i, b in enumerate(boxes):
        s.box(*b, size=16)
        if i:
            s.arrow(boxes[i - 1][0] + boxes[i - 1][2], 128, b[0], 128)
    s.note(220, 200, "same checkpoint every time   ·   miss contract is explicit", fill=SKY)
    s.write()


def stack() -> None:
    s = Scene("stack", 760, 390, "one arena, many layers")
    layers = [
        (70, 58, 620, 52, PINK, "C  /  Python  /  LeRobot plugin"),
        (70, 118, 620, 52, SUN, "runtime   ·   fe_engine_sample"),
        (70, 178, 300, 52, TEAL, "Mamba backbone"),
        (390, 178, 300, 52, PURPLE, "flow / DP / expert head"),
        (70, 238, 620, 52, ORANGE, "kernels.h    AVX2  /  NEON  /  scalar"),
        (70, 298, 620, 52, GREEN, "arena slab    weights + scratch    no malloc"),
    ]
    for b in layers:
        s.box(*b, size=16)
    s.write()


def arena() -> None:
    s = Scene("arena", 900, 250, "the whole allocator")
    s.box(36, 88, 828, 100, CREAM, "")
    s.box(56, 104, 190, 68, TEAL, "weights")
    s.box(266, 104, 200, 68, PURPLE, "decode / ODE")
    s.box(486, 104, 160, 68, ORANGE, "scratch")
    s.box(666, 104, 170, 68, PINK, "thread pool")
    s.arrow(566, 70, 566, 104, color=CORAL)
    s.caption(580, 64, "reset_to(mark)  rewind, reuse bytes")
    s.note(36, 44, "sized at load  ·  64-byte align  ·  fail closed")
    s.star(850, 48, 8, CORAL)
    s.write()


def mamba() -> None:
    s = Scene("mamba", 980, 240, "Mamba layer   ·   selective SSM")
    labels = [
        (16, 78, 112, 72, SUN, "RMSNorm"),
        (144, 78, 100, 72, TEAL, "in_proj"),
        (260, 78, 100, 72, PINK, "conv1d"),
        (376, 78, 88, 72, ORANGE, "SiLU"),
        (480, 78, 160, 72, PURPLE, "scan|A, B, C"),
        (656, 78, 110, 72, GREEN, "gate"),
        (782, 78, 170, 72, SUN, "out + res"),
    ]
    for i, b in enumerate(labels):
        s.box(*b, size=13)
        if i:
            s.arrow(labels[i - 1][0] + labels[i - 1][2], 114, b[0], 114)
    s.note(340, 175, "h_t = Ā h_{t-1} + B̄ x_t     state does not grow with time", fill=SKY)
    s.write()


def kernels() -> None:
    s = Scene("kernels", 760, 280, "one header, many ISAs")
    s.box(250, 54, 260, 64, SUN, "kernels.h")
    s.box(36, 168, 200, 68, TEAL, "AVX2")
    s.box(280, 168, 200, 68, GREEN, "NEON")
    s.box(524, 168, 200, 68, PINK, "scalar")
    s.arrow(310, 118, 136, 168)
    s.arrow(380, 118, 380, 168)
    s.arrow(450, 118, 624, 168)
    s.note(80, 248, "CUDA flow/DP heads resident; Mamba copies spans; Vulkan / Tenstorrent planned", fill=CREAM)
    s.star(70, 70, 8, ORANGE)
    s.write()


def relay() -> None:
    s = Scene("relay", 1180, 400, "optional Relay flow")

    s.caption(36, 64, "across processes", size=15)
    s.box(24, 78, 190, 86, SUN, "sensor / VLA")
    s.box(246, 78, 210, 86, PINK, "condition ring|checksum SPSC")
    s.box(488, 78, 210, 86, CORAL, "flowedge-relayd")
    s.box(730, 78, 200, 86, TEAL, "Core flow head")
    s.box(962, 78, 190, 86, GREEN, "action ring")
    s.arrow(214, 121, 246, 121)
    s.arrow(456, 121, 488, 121)
    s.arrow(698, 121, 730, 121)
    s.arrow(930, 121, 962, 121)

    s.caption(36, 194, "then the same period contract", size=15)
    s.box(24, 210, 190, 86, PINK, "controller")
    s.box(246, 210, 170, 86, SUN, "period?")
    s.box(448, 210, 230, 86, ORANGE, "on-miss|hold / drop / raise")
    s.box(710, 210, 190, 86, GREEN, "motors")
    s.arrow(214, 253, 246, 253)
    s.arrow(416, 253, 448, 253)
    s.arrow(678, 253, 710, 253)
    s.doodle_clock(980, 236, 18)

    s.note(36, 314, "DP replay does not start this", fill=ORANGE)
    s.caption(
        590,
        372,
        "local shared memory  ·  Core has no Relay dependency  ·  same --on-miss",
        size=14,
        anchor="middle",
    )
    s.write()


def deadline() -> None:
    s = Scene("deadline", 920, 270, "the period is the product")
    s.oval(40, 90, 150, 120, SUN, "10 ms")
    s.box(250, 100, 260, 96, CORAL, "sample  ~851 ms|CPU DDIM p50")
    s.box(560, 100, 310, 96, GREEN, "on-miss  hold|last action stays")
    s.arrow(190, 150, 250, 148)
    s.arrow(510, 148, 560, 148)
    s.note(180, 220, "faster than PyTorch here  ≠  a 100 Hz loop", fill=SKY)
    s.write()


def verify() -> None:
    s = Scene("verify", 860, 250, "same checkpoint, two engines")
    s.box(310, 44, 240, 54, SUN, ".safetensors")
    s.box(50, 140, 250, 72, TEAL, "FlowEdge")
    s.box(560, 140, 250, 72, PINK, "PyTorch ref")
    s.box(280, 175, 300, 44, ORANGE, "ULP  +  rel-error")
    s.arrow(370, 98, 175, 140)
    s.arrow(490, 98, 685, 140)
    s.write()


def convert() -> None:
    s = Scene("convert", 900, 220, "convert once")
    s.box(24, 78, 180, 86, SUN, "HF / torch")
    s.box(248, 78, 160, 86, PINK, "map names")
    s.box(452, 78, 170, 86, ORANGE, "validate")
    s.box(668, 78, 200, 86, TEAL, "Engine")
    s.arrow(204, 121, 248, 121)
    s.arrow(408, 121, 452, 121)
    s.arrow(622, 121, 668, 121)
    s.note(300, 178, "layouts the arena already knows", fill=SKY)
    s.write()


def benches() -> None:
    s = Scene("benches", 940, 250, "do not mix the evidence")
    s.box(24, 78, 200, 96, SUN, "kernels|microbench")
    s.box(252, 78, 200, 96, PINK, "Relay tail|queues")
    s.box(480, 78, 200, 96, TEAL, "policy vs PT|matched replay")
    s.box(708, 78, 200, 96, GREEN, "period misses|robot loop")
    s.note(220, 196, "only the last two are control-loop numbers", fill=ORANGE)
    s.write()


def delivery() -> None:
    s = Scene("delivery", 940, 210, "action delivery gate")
    s.box(16, 74, 150, 76, SUN, "chunk")
    s.box(196, 74, 160, 76, PINK, "freshness")
    s.box(386, 74, 150, 76, ORANGE, "overlap")
    s.box(566, 74, 150, 76, TEAL, "bounds")
    s.box(746, 74, 160, 76, GREEN, "motor")
    for x in (166, 356, 536, 716):
        s.arrow(x, 112, x + 30, 112)
    s.write()


def jobs() -> None:
    s = Scene("jobs", 900, 240, "cooperative jobs")
    s.box(24, 80, 170, 86, SUN, "JobClient")
    s.box(234, 80, 190, 86, PINK, "shared rings")
    s.box(464, 80, 170, 86, ORANGE, "EDF pool")
    s.box(674, 80, 190, 86, TEAL, "Mamba lane")
    s.arrow(194, 123, 234, 123)
    s.arrow(424, 123, 464, 123)
    s.arrow(634, 123, 674, 123)
    s.note(280, 186, "one token = one work unit", fill=SKY)
    s.write()


def coop_flow() -> None:
    s = Scene("coop-flow", 860, 250, "split the ODE, not the matmul")
    s.box(30, 90, 220, 86, SUN, "flow_begin")
    s.box(300, 90, 250, 86, PURPLE, "flow_advance(N)")
    s.box(600, 90, 230, 86, GREEN, "publish at NFE=0")
    s.arrow(250, 133, 300, 133)
    s.arrow(550, 133, 600, 133)
    s.note(200, 196, "bit-identical to one-shot Euler / Heun / RK4", fill=ORANGE)
    s.write()


def observe() -> None:
    s = Scene("observe", 860, 230, "record off the hot path")
    s.box(30, 90, 190, 76, TEAL, "worker pool")
    s.box(270, 90, 220, 76, SUN, "200-byte events")
    s.box(560, 54, 260, 58, PINK, "metrics")
    s.box(560, 132, 260, 58, ORANGE, "trace JSONL")
    s.arrow(220, 128, 270, 128)
    s.arrow(490, 110, 560, 82, dashed=True)
    s.arrow(490, 140, 560, 160, dashed=True)
    s.write()


def profile() -> None:
    s = Scene("profile", 860, 210, "deadline profile")
    s.box(24, 74, 160, 76, SUN, "warmup")
    s.box(220, 74, 180, 76, TEAL, "timed loop")
    s.box(436, 74, 170, 76, ORANGE, "p50…p999")
    s.box(642, 74, 190, 76, GREEN, "pass / CI fail")
    s.arrow(184, 112, 220, 112)
    s.arrow(400, 112, 436, 112)
    s.arrow(606, 112, 642, 112)
    s.write()


def sequence() -> None:
    s = Scene("sequence", 940, 210, "what we ship next")
    s.box(16, 74, 170, 76, SUN, "convert")
    s.box(216, 74, 190, 76, TEAL, "CPU replay")
    s.box(436, 74, 220, 76, ORANGE, "Jetson / SO-100")
    s.box(686, 74, 230, 76, GREEN, "same contracts")
    s.arrow(186, 112, 216, 112)
    s.arrow(406, 112, 436, 112)
    s.arrow(656, 112, 686, 112)
    s.write()


def dataflow() -> None:
    s = Scene("dataflow", 900, 250, "one sample")
    s.box(20, 86, 150, 86, SUN, "tokens|or condition")
    s.box(210, 86, 160, 86, TEAL, "backbone")
    s.box(410, 86, 200, 86, PURPLE, "flow ODE|x0 → x1")
    s.box(650, 86, 220, 86, GREEN, "action chunk")
    s.arrow(170, 129, 210, 129)
    s.arrow(370, 129, 410, 129)
    s.arrow(610, 129, 650, 129)
    s.note(250, 190, "caller may skip the backbone and hand in c", fill=SKY)
    s.write()


def matrix() -> None:
    s = Scene("matrix", 860, 270, "two product paths")
    s.box(40, 70, 200, 80, TEAL, "Mamba")
    s.box(300, 70, 240, 80, PURPLE, "flow matching")
    s.arrow(240, 110, 300, 110)
    s.box(40, 170, 200, 80, PINK, "LeRobot encoder")
    s.box(300, 170, 240, 80, ORANGE, "DP U-Net")
    s.arrow(240, 210, 300, 210)
    s.oval(620, 100, 200, 100, GREEN, "action chunk")
    s.arrow(540, 110, 620, 140)
    s.arrow(540, 210, 620, 170)
    s.note(560, 40, "Transformer / π0 / DiT: same contract", fill=SUN)
    s.write()


def compare() -> None:
    s = Scene("compare", 920, 340, "matched PushT  ·  policy p50  ·  threads=1")
    s.box(40, 80, 200, 70, PINK, "PyTorch")
    s.box(40, 170, 200, 70, TEAL, "FlowEdge")
    s.svg.append(
        f'<rect x="260" y="92" width="396" height="46" rx="10" fill="{CORAL}" stroke="{INK}" stroke-width="2.4"/>'
        f'<text x="470" y="122" text-anchor="middle" font-size="18" '
        f'font-family="Trebuchet MS, Segoe UI, sans-serif" font-weight="800" fill="{INK}">1409 ms</text>'
        f'<rect x="260" y="182" width="238" height="46" rx="10" fill="{GREEN}" stroke="{INK}" stroke-width="2.4"/>'
        f'<text x="379" y="212" text-anchor="middle" font-size="18" '
        f'font-family="Trebuchet MS, Segoe UI, sans-serif" font-weight="800" fill="{INK}">851 ms</text>'
    )
    s.note(620, 88, "0.60×  ·  1.66× faster", fill=SUN)
    s.note(620, 178, "same checkpoint + DDIM", fill=SKY)
    s.caption(460, 280, "CPU  ·  threads=1  ·  not a 10 ms loop", size=14, anchor="middle")
    s.caption(460, 308, "threads=2: 800 vs 1031 ms    threads=4: 633 vs 819 ms", size=14, anchor="middle")
    s.write()


def vs_arch() -> None:
    s = Scene("vs-arch", 980, 280, "same checkpoint, two engines")
    s.box(50, 70, 420, 150, TEAL, "")
    s.caption(260, 108, "Flow matching", size=20, anchor="middle")
    s.lines(260, 140, ["vs PyTorch ULP on mamba_flow", "~1e-6 rel   ·   Euler NFE=10", "correctness, not a policy p50"], size=14)

    s.box(510, 70, 420, 150, GREEN, "")
    s.caption(720, 108, "Diffusion Policy", size=20, anchor="middle")
    s.lines(720, 140, ["vs LeRobot/PyTorch DDIM", "851 vs 1409 ms policy p50", "threads=1, matched PushT"], size=14)

    s.caption(490, 250, "two heads, two claims  ·  do not average them into one speedup", size=14, anchor="middle")
    s.write()


def policies() -> None:
    s = Scene("policies", 980, 340, "two ways to turn noise into an action")
    s.box(36, 70, 440, 210, TEAL, "")
    s.caption(256, 108, "Flow matching", size=22, anchor="middle")
    s.lines(
        256,
        142,
        [
            "dx/dt = v(x, t | c)",
            "x0 noise  →  x1 action",
            "Euler / Heun / RK4   ·   short NFE",
            "c from Mamba, or your encoder",
        ],
        size=15,
    )

    s.box(504, 70, 440, 210, GREEN, "")
    s.caption(724, 108, "Diffusion Policy", size=22, anchor="middle")
    s.lines(
        724,
        142,
        [
            "epsilon U-Net on a noisy horizon",
            "x_T noise  →  x0 action chunk",
            "DDIM (or seeded DDPM)   ·   more NFE",
            "c from LeRobot RGB encoder",
        ],
        size=15,
    )

    s.caption(490, 310, "same arena, same miss contract  ·  pick the head you trained", size=14, anchor="middle")
    s.write()


def big_flow() -> None:
    s = Scene("big-flow", 1180, 460, "the whole FlowEdge flow")

    s.box(24, 58, 170, 72, SUN, "HF / LeRobot|checkpoint")
    s.box(230, 58, 140, 72, PINK, "convert")
    s.box(406, 58, 200, 72, ORANGE, ".safetensors")
    s.box(642, 58, 200, 72, TEAL, "load arena|fixed RSS")
    s.arrow(194, 94, 230, 94)
    s.arrow(370, 94, 406, 94)
    s.arrow(606, 94, 642, 94)
    s.note(860, 70, "malloc = 0 after load", fill=SKY)

    s.caption(36, 160, "flow matching", size=16)
    s.box(24, 176, 150, 68, SUN, "prefix")
    s.box(198, 176, 160, 68, TEAL, "Mamba")
    s.box(382, 176, 70, 68, CREAM, "c")
    s.box(476, 176, 230, 68, PURPLE, "flow ODE|Euler / Heun / RK4")
    s.arrow(174, 210, 198, 210)
    s.arrow(358, 210, 382, 210)
    s.arrow(452, 210, 476, 210)

    s.caption(36, 268, "Diffusion Policy", size=16)
    s.box(24, 284, 150, 68, PINK, "RGB + state")
    s.box(198, 284, 160, 68, CORAL, "LeRobot encoder")
    s.box(382, 284, 70, 68, CREAM, "c")
    s.box(476, 284, 230, 68, ORANGE, "DP U-Net|DDIM / DDPM")
    s.arrow(174, 318, 198, 318)
    s.arrow(358, 318, 382, 318)
    s.arrow(452, 318, 476, 318)

    s.box(760, 210, 190, 100, GREEN, "action chunk")
    s.arrow(706, 210, 760, 240)
    s.arrow(706, 318, 760, 280)

    s.box(980, 150, 170, 64, SUN, "period?")
    s.box(980, 230, 170, 64, ORANGE, "on-miss|hold / drop / raise")
    s.box(980, 310, 170, 64, GREEN, "motors")
    s.arrow(950, 250, 980, 182)
    s.arrow(1065, 214, 1065, 230)
    s.arrow(1065, 294, 1065, 310)
    s.doodle_clock(1128, 118, 16)

    s.caption(590, 430, "same checkpoint in   ·   action out   ·   encoder stays outside Core", size=14, anchor="middle")
    s.write()


def _fonts():
    try:
        from PIL import ImageFont
    except ImportError:
        return None, None, None, None
    for bold, regular in (
        ("C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/segoeui.ttf"),
        ("C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf"),
    ):
        try:
            return (
                ImageFont.truetype(bold, 26),
                ImageFont.truetype(regular, 20),
                ImageFont.truetype(regular, 15),
                ImageFont.truetype(bold, 22),
            )
        except OSError:
            continue
    d = ImageFont.load_default()
    return d, d, d, d


def _gif() -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("skip gif: pillow missing")
        return

    big, font, small, _ = _fonts()
    w, h = 760, 300
    n = 16
    frames = []
    cream = (255, 246, 220)
    ink = (43, 33, 24)
    navy = (27, 75, 107)
    coral = (255, 107, 107)
    sun = (255, 209, 102)
    green = (107, 203, 119)
    purple = (123, 108, 246)
    teal = (46, 196, 182)

    def pt(u: float) -> tuple[float, float]:
        return 80 + u * 560, 155 + 58 * math.sin(u * math.pi)

    for i in range(n):
        t = i / (n - 1)
        img = Image.new("RGB", (w, h), cream)
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((7, 7, w - 8, h - 8), 20, outline=(232, 215, 168), width=4)
        d.text((24, 14), "flow matching", font=big, fill=navy)
        d.text((250, 20), "dx/dt = v(x, t | c)", font=font, fill=ink)
        curve = [pt(u / 80) for u in range(81)]
        d.line(curve, fill=coral, width=6)
        d.ellipse((48, 118, 122, 192), fill=sun, outline=ink, width=3)
        d.text((60, 200), "noise x0", font=small, fill=ink)
        d.ellipse((638, 110, 732, 204), fill=green, outline=ink, width=3)
        d.text((650, 214), "action x1", font=small, fill=ink)
        for k in range(i + 1):
            u = k / (n - 1)
            x, y = pt(u)
            d.ellipse((x - 6, y - 6, x + 6, y + 6), fill=purple, outline=ink, width=2)
        x, y = pt(t)
        d.ellipse((x - 16, y - 16, x + 16, y + 16), fill=purple, outline=ink, width=3)
        d.rounded_rectangle((200, 228, 560, 272), 12, fill=teal, outline=ink, width=3)
        d.text((218, 238), f"Euler  t={t:.2f}    malloc = 0    NFE++", font=font, fill=ink)
        frames.append(img)

    hold = frames[-1].copy()
    frames.extend([hold] * 4)
    out = HERE.parent / "flowedge.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=160, loop=0, optimize=True)
    print(f"wrote {out}")


def _perf_gif() -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return

    coral = (255, 107, 107)
    teal = (46, 196, 182)
    green = (107, 203, 119)
    pink = (255, 143, 171)
    groups = (
        ("CUDA    GTX 1650", "131 vs 345 ms", (
            ("PyTorch CUDA", 345, coral, pink),
            ("FlowEdge CUDA", 131, green, teal),
        )),
        ("CPU    threads=1", "851 vs 1409 ms", (
            ("PyTorch CPU", 1409, coral, pink),
            ("FlowEdge CPU", 851, green, teal),
        )),
    )

    big, _font, small, mid = _fonts()
    w, h = 760, 540
    cream = (255, 246, 220)
    track = (245, 232, 200)
    ink = (43, 33, 24)
    navy = (27, 75, 107)
    sun = (255, 209, 102)
    white = (255, 254, 248)
    bar_left, bar_right = 210, 736
    bar_w = bar_right - bar_left
    bar_h = 40

    n = 36
    frames = []
    for i in range(n):
        sample_t = 0.0 if i < 10 else min(1.0, (i - 10) / 18)
        img = Image.new("RGB", (w, h), cream)
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((7, 7, w - 8, h - 8), 20, outline=(232, 215, 168), width=4)
        d.text((24, 14), "Matched replay", font=big, fill=navy)
        d.text((24, 48), "same checkpoint / DDIM / observations    load → warmup → sample", font=small, fill=ink)

        if i < 5:
            status = "loading checkpoint into arena"
        elif i < 10:
            status = "warmup    malloc after warmup: 0"
        elif sample_t < 1.0:
            status = "sampling DDIM" + ("." * (1 + (i % 3)))
        else:
            status = "policy p50 locked    malloc after warmup: 0"
        d.rounded_rectangle((24, 72, 736, 112), 12, fill=sun if sample_t >= 1.0 else white, outline=ink, width=3)
        d.text((40, 82), status, font=mid, fill=ink)

        y = 126
        for g, (heading, pair, rows) in enumerate(groups):
            d.text((24, y), heading, font=mid, fill=navy)
            d.text((520, y), pair, font=small, fill=ink)
            y += 30
            slowest = max(row[1] for row in rows)
            for r, (label, ms, fill, chip) in enumerate(rows):
                d.rounded_rectangle((24, y, 196, y + bar_h), 12, fill=chip, outline=ink, width=3)
                d.text((36, y + 10), label, font=small, fill=ink)
                d.rounded_rectangle((bar_left, y, bar_right, y + bar_h), 12, fill=track, outline=ink, width=3)
                if i < 10:
                    pulse = 12 + int(18 * (0.5 + 0.5 * math.sin(i * 0.9 + g * 2 + r)))
                    d.rounded_rectangle((bar_left, y, bar_left + pulse, y + bar_h), 12, fill=fill, outline=ink, width=3)
                    d.text((bar_left + 12, y + 10), "waiting", font=small, fill=ink)
                else:
                    duration = ms / slowest
                    frac = min(1.0, sample_t / max(duration, 0.05))
                    pw = max(8, int(bar_w * frac))
                    d.rounded_rectangle((bar_left, y, bar_left + pw, y + bar_h), 12, fill=fill, outline=ink, width=3)
                    d.text((bar_left + 12, y + 8), f"{int(ms * frac)} ms", font=mid, fill=ink)
                y += 54
            y += 10

        d.text((24, 478), "each pair is its own race    first bar to 100% in that pair is faster", font=small, fill=ink)
        d.text((24, 504), "not a 10 ms loop    not Jetson/ARM    in-process Engine", font=small, fill=navy)
        frames.append(img)

    hold = [frames[-1]] * 12
    loop = frames + hold
    README_ASSETS.mkdir(parents=True, exist_ok=True)
    out = README_ASSETS / "matched_replay.gif"
    loop[0].save(out, save_all=True, append_images=loop[1:], duration=90, loop=0, optimize=True)
    (HERE.parent / "matched_replay.gif").write_bytes(out.read_bytes())
    print(f"wrote {out}")


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    for fn in (
        purpose,
        flow_matching,
        workflow,
        stack,
        arena,
        mamba,
        kernels,
        relay,
        deadline,
        verify,
        convert,
        benches,
        delivery,
        jobs,
        coop_flow,
        observe,
        profile,
        sequence,
        dataflow,
        matrix,
        compare,
        vs_arch,
        policies,
        big_flow,
    ):
        fn()
    _gif()
    _perf_gif()
    print(f"wrote figures in {HERE}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "perf":
        _perf_gif()
    else:
        main()
