import argparse
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas


OUT = Path(__file__).resolve().parent / "figures"

BLUE = HexColor("#2A6F97")
LIGHT_BLUE = HexColor("#61A5C2")
ORANGE = HexColor("#D4A373")
GREEN = HexColor("#2A9D8F")
RED = HexColor("#C44900")
INK = HexColor("#243B53")
GRID = HexColor("#D8E0E7")


def label(c, x, y, text, size=8, color=INK, anchor="middle"):
    c.setFont("Helvetica", size)
    c.setFillColor(color)
    if anchor == "middle":
        c.drawCentredString(x, y, text)
    elif anchor == "end":
        c.drawRightString(x, y, text)
    else:
        c.drawString(x, y, text)


def overall_tradeoff():
    c = canvas.Canvas(str(OUT / "overall_tradeoff.pdf"), pagesize=(510, 205))
    label(c, 130, 188, "Computational reduction", 10)
    label(c, 385, 188, "Threshold-relevant accuracy", 10)
    for y in (40, 75, 110, 145):
        c.setStrokeColor(GRID)
        c.line(36, y, 232, y)
    vals = [(35.52, 26.89), (14.08, 7.73)]
    xs = [94, 184]
    for x, (calls, runtime) in zip(xs, vals):
        c.setFillColor(BLUE)
        c.rect(x - 20, 32, 17, calls * 3.2, fill=1, stroke=0)
        c.setFillColor(LIGHT_BLUE)
        c.rect(x + 3, 32, 17, runtime * 3.2, fill=1, stroke=0)
    label(c, xs[0], 18, "Held-out NB", 8)
    label(c, xs[1], 18, "MI", 8)
    label(c, 38, 157, "Reduction (%)", 8, anchor="start")
    c.setFillColor(BLUE)
    c.rect(49, 171, 8, 8, fill=1, stroke=0)
    label(c, 61, 171, "Calls", 7, anchor="start")
    c.setFillColor(LIGHT_BLUE)
    c.rect(101, 171, 8, 8, fill=1, stroke=0)
    label(c, 113, 171, "Time", 7, anchor="start")

    c.setStrokeColor(INK)
    c.line(280, 104, 488, 104)
    changes = [18.45, -46.18]
    for x, v in zip([340, 430], changes):
        c.setFillColor(RED if v > 0 else GREEN)
        y = 104 if v > 0 else 104 + v * 1.35
        c.rect(x - 15, min(104, y), 30, abs(v) * 1.35, fill=1, stroke=0)
        label(c, x, 135 if v > 0 else 31, f"{v:+.1f}%", 8)
    label(c, 340, 18, "Held-out NB", 8)
    label(c, 430, 18, "MI", 8)
    label(c, 290, 157, "High-NMAE change (%)", 8, anchor="start")
    c.save()


def method_overview():
    c = canvas.Canvas(str(OUT / "method_overview.pdf"), pagesize=(510, 155))
    boxes = [
        (15, ["Full-sampling", "warm-up"]),
        (115, ["Estimate mean, SE,", "and context coverage"]),
        (225, ["Dynamic high / uncertain /", "low classification"]),
        (345, ["Assign predictable", "inclusion probabilities"]),
        (445, ["HT update, audit,", "and fallback"]),
    ]
    widths = [78, 94, 105, 90, 60]
    for (x, lines), width in zip(boxes, widths):
        c.setFillColor(HexColor("#EAF2F8"))
        c.setStrokeColor(BLUE)
        c.roundRect(x, 68, width, 42, 4, fill=1, stroke=1)
        for j, line in enumerate(lines):
            label(c, x + width / 2, 92 - j * 12, line, 7.5)
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + widths[i]
        x2 = boxes[i + 1][0]
        c.setStrokeColor(INK)
        c.line(x1 + 2, 89, x2 - 5, 89)
        c.line(x2 - 10, 93, x2 - 5, 89)
        c.line(x2 - 10, 85, x2 - 5, 89)
    c.setStrokeColor(RED)
    c.bezier(470, 62, 430, 20, 290, 18, 280, 62)
    c.line(276, 54, 280, 62)
    c.line(286, 57, 280, 62)
    label(c, 370, 14, "periodic regrouping", 7, RED)
    c.save()


def low_tail_scaling():
    c = canvas.Canvas(str(OUT / "low_tail_scaling.pdf"), pagesize=(510, 250))
    counts = [8, 16, 32]
    call_reduction = [19.2, 42.1, 54.7]
    low_recall = [17.0, 56.0, 98.0]
    xs = [105, 255, 405]
    baseline = 42
    scale = 1.65

    label(c, 255, 232, "Benefit increases with the size of the low-contribution tail", 10)
    for step in range(0, 101, 20):
        y = baseline + step * scale
        c.setStrokeColor(GRID)
        c.line(48, y, 470, y)
        label(c, 42, y - 3, str(step), 7, anchor="end")
    label(c, 48, 198, "Percent", 7, anchor="start")

    c.setLineWidth(2.2)
    c.setStrokeColor(BLUE)
    for index in range(len(xs) - 1):
        c.line(
            xs[index], baseline + call_reduction[index] * scale,
            xs[index + 1], baseline + call_reduction[index + 1] * scale,
        )
    c.setStrokeColor(ORANGE)
    for index in range(len(xs) - 1):
        c.line(
            xs[index], baseline + low_recall[index] * scale,
            xs[index + 1], baseline + low_recall[index + 1] * scale,
        )
    for x, count, calls, recall in zip(xs, counts, call_reduction, low_recall):
        c.setFillColor(BLUE)
        c.circle(x, baseline + calls * scale, 4, fill=1, stroke=0)
        label(c, x - 8, baseline + calls * scale + 8, f"{calls:.1f}%", 7, BLUE, "end")
        c.setFillColor(ORANGE)
        c.circle(x, baseline + recall * scale, 4, fill=1, stroke=0)
        label(c, x + 8, baseline + recall * scale + 8, f"{recall:.0f}%", 7, ORANGE, "start")
        label(c, x, 22, str(count), 8)

    c.setFillColor(BLUE)
    c.rect(116, 207, 8, 8, fill=1, stroke=0)
    label(c, 129, 207, "Actual-call reduction vs MC", 7, anchor="start")
    c.setFillColor(ORANGE)
    c.rect(300, 207, 8, 8, fill=1, stroke=0)
    label(c, 313, 207, "True-low recall", 7, anchor="start")
    label(c, 255, 7, "Number of individually low-contribution members", 8)
    c.save()


def main():
    global OUT
    parser = argparse.ArgumentParser(description="Generate the manuscript figures.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT,
        help="Directory for the three generated PDF figures.",
    )
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    overall_tradeoff()
    method_overview()
    low_tail_scaling()


if __name__ == "__main__":
    main()
