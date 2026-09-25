"""Render the editable Figure 7 PPTX with native PowerPoint and Poppler."""

from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path

import win32com.client as win32


BASE = Path(__file__).resolve().parents[1] / "figure7_q3_framework_zh_v2"


def main() -> None:
    if not ctypes.windll.user32.GetDesktopWindow():
        raise RuntimeError("PowerPoint export requires an interactive Windows desktop session")
    pptx = BASE.with_suffix(".pptx")
    if not pptx.exists():
        raise FileNotFoundError(pptx)
    application = win32.DispatchEx("PowerPoint.Application")
    presentation = None
    try:
        presentation = application.Presentations.Open(str(pptx), ReadOnly=True, WithWindow=False)
        if presentation.Slides.Count != 1:
            raise ValueError("Expected a one-slide Figure 7 deck")
        presentation.Slides(1).Export(str(BASE.with_suffix(".png")), "PNG", 4000, 2250)
        presentation.SaveAs(str(BASE.with_suffix(".pdf")), 32)  # ppSaveAsPDF
    finally:
        if presentation is not None:
            presentation.Close()
        application.Quit()
    subprocess.run(["pdftocairo", "-svg", str(BASE.with_suffix(".pdf")),
                    str(BASE.with_suffix(".svg"))], check=True)
    print(f"EXPORTED {BASE}.png/.pdf/.svg")


if __name__ == "__main__":
    main()
