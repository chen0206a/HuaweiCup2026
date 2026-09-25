"""Export the Q2-layout/Q3-content PowerPoint draft for manuscript review."""

from pathlib import Path
import subprocess

import win32com.client as win32


HERE = Path(__file__).resolve().parent
PPTX = HERE / "q2_framework_q3_draft.pptx"
PNG = HERE / "figure7_q3_into_q2_layout.png"
PDF = HERE / "figure7_q3_into_q2_layout.pdf"
SVG = HERE / "figure7_q3_into_q2_layout.svg"


def main() -> None:
    app = win32.DispatchEx("PowerPoint.Application")
    try:
        presentation = app.Presentations.Open(str(PPTX), ReadOnly=True, WithWindow=False)
        try:
            assert presentation.Slides.Count == 1
            presentation.Slides(1).Export(str(PNG), "PNG", 4000, 2250)
            presentation.SaveAs(str(PDF), 32)
        finally:
            presentation.Close()
    finally:
        app.Quit()
    subprocess.run(["pdftocairo", "-svg", str(PDF), str(SVG)], check=True)
    print(PNG)
    print(PDF)
    print(SVG)


if __name__ == "__main__":
    main()
