"""Generate the images used by the multimodal evaluation scenarios.

The pages are rendered from the indexed text itself, so the scenarios stay
reproducible without shipping scans: python scripts/make_scenario_images.py
"""
from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/annotations/scenario_images"


def font(size: int, serif: bool = True):
    for name in (["georgia.ttf", "times.ttf", "DejaVuSerif.ttf"] if serif else ["arialbd.ttf", "DejaVuSans-Bold.ttf"]):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def passage(work_id: str, start: str, length: int = 900) -> str:
    chunks = json.loads((ROOT / f"data/processed/{work_id}.chunks.json").read_text(encoding="utf-8"))
    for chunk in chunks:
        text = re.sub(r"\s+", " ", chunk["text"])
        index = text.find(start)
        if index >= 0:
            return text[index:index + length].rsplit(" ", 1)[0] + "…"
    raise SystemExit(f"Passage introuvable : {start}")


def page(name: str, heading: str, body: str) -> None:
    image = Image.new("RGB", (1000, 1300), "#f6f1e4")
    draw = ImageDraw.Draw(image)
    draw.text((90, 90), heading, fill="#3b3227", font=font(38))
    draw.line((90, 150, 910, 150), fill="#b9ab90", width=2)
    y = 190
    for line in textwrap.wrap(body, 52):
        draw.text((90, y), line, fill="#2b241c", font=font(30))
        y += 46
    image.save(OUT / name)


def violin(name: str) -> None:
    image = Image.new("RGB", (900, 900), "#fbf8f1")
    draw = ImageDraw.Draw(image)
    body = "#9b5a2a"
    draw.ellipse((300, 470, 600, 760), fill=body)
    draw.ellipse((330, 300, 570, 520), fill=body)
    draw.rectangle((395, 470, 505, 560), fill=body)
    draw.rectangle((432, 110, 468, 330), fill="#2d1d12")
    draw.ellipse((418, 70, 482, 130), fill="#2d1d12")
    for x in (440, 450, 460):
        draw.line((x, 120, x, 690), fill="#e8e0cf", width=2)
    draw.arc((345, 520, 405, 600), 270, 90, fill="#2d1d12", width=6)
    draw.arc((495, 520, 555, 600), 90, 270, fill="#2d1d12", width=6)
    draw.line((180, 780, 760, 300), fill="#6b4a2f", width=8)
    draw.text((330, 800), "VIOLIN", fill="#2d1d12", font=font(64, serif=False))
    image.save(OUT / name)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    page("metamorphosis_opening.png", "I", passage("metamorphosis", "One morning, when Gregor Samsa woke"))
    page("trial_doorkeeper.png", "In the Cathedral", passage("the_trial", "In front of the law there is a doorkeeper", 700))
    violin("violin_card.png")
    print(f"Images écrites dans {OUT}")
