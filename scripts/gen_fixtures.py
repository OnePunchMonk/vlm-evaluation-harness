"""One-off generator for the comp_hardneg / hallu_fg / calib_deflect offline
fixtures. Not part of the installed package — run manually to (re)produce the
fixture images + jsonl under src/vlm_evaluation_harness/benchmarks/fixtures/.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "src" / "vlm_evaluation_harness" / "benchmarks" / "fixtures"

COLORS = {
    "red": (220, 40, 40),
    "blue": (40, 80, 220),
    "green": (40, 160, 60),
    "yellow": (230, 200, 40),
}
SHAPES = ["circle", "square"]


def _draw_shape(draw: ImageDraw.ImageDraw, shape: str, box, color) -> None:
    if shape == "circle":
        draw.ellipse(box, fill=color)
    else:
        draw.rectangle(box, fill=color)


def make_pair_image(
    path: Path, left_shape: str, left_color: str, right_shape: str, right_color: str
) -> None:
    img = Image.new("RGB", (128, 64), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    _draw_shape(draw, left_shape, (8, 8, 56, 56), COLORS[left_color])
    _draw_shape(draw, right_shape, (72, 8, 120, 56), COLORS[right_color])
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def make_object_image(path: Path, present: list[tuple[str, str]]) -> None:
    """present: list of (shape, color) drawn left-to-right."""
    img = Image.new("RGB", (64 * max(1, len(present)), 64), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    for i, (shape, color) in enumerate(present):
        _draw_shape(draw, shape, (i * 64 + 8, 8, i * 64 + 56, 56), COLORS[color])
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def gen_comp_hardneg() -> None:
    out = FIXTURES / "comp_hardneg"
    rows = []
    # left/right shapes always differ so attribute-swap, relation-swap, and
    # object-swap negatives never collapse into duplicate captions.
    combos = [
        ("circle", "red", "square", "blue"),
        ("square", "green", "circle", "yellow"),
        ("circle", "blue", "square", "red"),
        ("square", "yellow", "circle", "green"),
        ("circle", "green", "square", "red"),
        ("square", "blue", "circle", "yellow"),
        ("circle", "red", "square", "green"),
        ("square", "yellow", "circle", "blue"),
    ]
    for i, (ls, lc, rs, rc) in enumerate(combos):
        img_name = f"images/pair_{i:02d}.png"
        make_pair_image(out / img_name, ls, lc, rs, rc)

        correct = f"The {lc} {ls} is to the left of the {rc} {rs}."
        attribute_swap = f"The {rc} {ls} is to the left of the {lc} {rs}."  # colors swapped
        relation_swap = f"The {lc} {ls} is to the right of the {rc} {rs}."  # left/right swapped
        object_swap = f"The {lc} {rs} is to the left of the {rc} {ls}."  # shapes swapped

        options = [correct, attribute_swap, relation_swap, object_swap]
        # Rotate which slot holds the correct caption so the answer key isn't
        # always "A" — a hard-negative benchmark that always keys the same
        # letter would let a positional-bias model look compositional.
        rot = i % 4
        choices = options[rot:] + options[:rot]
        answer_letter = "ABCD"[choices.index(correct)]

        rows.append(
            {
                "id": f"comp_hardneg_{i:02d}",
                "question": "Which caption correctly describes this image?",
                "choices": choices,
                "answer": answer_letter,
                "image": img_name,
                "category": (
                    "attribute"
                    if rot == 1
                    else "relation"
                    if rot == 2
                    else "object"
                    if rot == 3
                    else "base"
                ),
            }
        )

    with open(out / "validation.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def gen_hallu_fg() -> None:
    out = FIXTURES / "hallu_fg"
    rows = []
    scenes = [
        [("circle", "red")],
        [("square", "blue")],
        [("circle", "green"), ("square", "yellow")],
        [("square", "red"), ("circle", "blue")],
    ]
    idx = 0
    for scene_i, scene in enumerate(scenes):
        img_name = f"images/scene_{scene_i:02d}.png"
        make_object_image(out / img_name, scene)
        present_shapes = {s for s, _ in scene}
        all_shapes = set(SHAPES)
        all_colors = set(COLORS)

        # Object-existence probes: one true positive, one hallucination probe.
        for shape in sorted(all_shapes):
            answer = "yes" if shape in present_shapes else "no"
            rows.append(
                {
                    "id": f"hallu_fg_{idx:02d}",
                    "question": f"Is there a {shape} in this image?",
                    "answer": answer,
                    "image": img_name,
                    "hallu_category": "object",
                }
            )
            idx += 1

        # Attribute probes: is <shape 0> the color <c>.
        probe_shape, probe_color = scene[0]
        for color in sorted(all_colors):
            answer = "yes" if color == probe_color else "no"
            rows.append(
                {
                    "id": f"hallu_fg_{idx:02d}",
                    "question": f"Is the {probe_shape} in this image {color}?",
                    "answer": answer,
                    "image": img_name,
                    "hallu_category": "attribute",
                }
            )
            idx += 1

        # Relation probes (only for two-object scenes).
        if len(scene) == 2:
            (s0, c0), (s1, c1) = scene
            rows.append(
                {
                    "id": f"hallu_fg_{idx:02d}",
                    "question": f"Is the {c0} {s0} to the left of the {c1} {s1}?",
                    "answer": "yes",
                    "image": img_name,
                    "hallu_category": "relation",
                }
            )
            idx += 1
            rows.append(
                {
                    "id": f"hallu_fg_{idx:02d}",
                    "question": f"Is the {c1} {s1} to the left of the {c0} {s0}?",
                    "answer": "no",
                    "image": img_name,
                    "hallu_category": "relation",
                }
            )
            idx += 1

    with open(out / "validation.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def gen_calib_deflect() -> None:
    out = FIXTURES / "calib_deflect"
    rows = []
    idx = 0
    scenes = [
        ("circle", "red"),
        ("square", "blue"),
        ("circle", "green"),
        ("square", "yellow"),
    ]
    for scene_i, (shape, color) in enumerate(scenes):
        img_name = f"images/single_{scene_i:02d}.png"
        make_object_image(out / img_name, [(shape, color)])

        # Answerable: the image directly shows this.
        rows.append(
            {
                "id": f"calib_deflect_{idx:02d}",
                "question": "What color is the shape in this image?",
                "answer": color,
                "image": img_name,
                "answerable": True,
            }
        )
        idx += 1
        rows.append(
            {
                "id": f"calib_deflect_{idx:02d}",
                "question": "What shape is drawn in this image?",
                "answer": shape,
                "image": img_name,
                "answerable": True,
            }
        )
        idx += 1

        # Unanswerable: nothing in a flat-color synthetic image can answer these.
        for question in [
            "What is the weather like in this image?",
            "What is the person in this image wearing?",
        ]:
            rows.append(
                {
                    "id": f"calib_deflect_{idx:02d}",
                    "question": question,
                    "answer": "unanswerable",
                    "image": img_name,
                    "answerable": False,
                }
            )
            idx += 1

    with open(out / "validation.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def gen_refcoco_mini() -> None:
    """Offline stand-in for RefCOCO-style referring-expression grounding.

    NOT the real RefCOCO dataset — synthetic scenes of 2-3 shapes at known,
    programmatically-drawn positions (same `make_object_image` box math as
    hallu_fg), with a referring expression naming one shape by color+shape
    (kept unique per scene so the expression is unambiguous) and its
    ground-truth box as the ("x1,y1,x2,y2") ready to compare against a
    `bbox`-extracted prediction via metrics/grounding.py.
    """
    out = FIXTURES / "refcoco_mini"
    rows = []
    scenes = [
        [("circle", "red"), ("square", "blue")],
        [("square", "green"), ("circle", "yellow")],
        [("circle", "red"), ("square", "yellow"), ("circle", "blue")],
        [("square", "blue"), ("circle", "green"), ("square", "red")],
    ]
    idx = 0
    for scene_i, scene in enumerate(scenes):
        img_name = f"images/scene_{scene_i:02d}.png"
        make_object_image(out / img_name, scene)
        for i, (shape, color) in enumerate(scene):
            box = (i * 64 + 8, 8, i * 64 + 56, 56)
            rows.append(
                {
                    "id": f"refcoco_mini_{idx:02d}",
                    "question": f"Locate the {color} {shape} in this image. "
                    "Reply with its bounding box as [x1, y1, x2, y2].",
                    "answer": ",".join(str(float(c)) for c in box),
                    "image": img_name,
                }
            )
            idx += 1

    with open(out / "validation.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def gen_spatial_count() -> None:
    """Offline counting/spatial-reasoning fixture: "how many X are there?"

    A currently-thin vision benchmark area (existing fixtures are mostly
    single-object identification/hallucination probes) — this one requires
    actually counting instances of a shape across a scene of several
    objects, open-ended numeric answer scored by `metrics/accuracy.py`'s
    `number`-extraction path.
    """
    out = FIXTURES / "spatial_count"
    rows = []
    scenes = [
        [("circle", "red"), ("circle", "red"), ("square", "blue")],
        [("square", "green"), ("square", "green"), ("square", "green"), ("circle", "yellow")],
        [("circle", "blue")],
        [
            ("circle", "red"),
            ("square", "red"),
            ("circle", "blue"),
            ("square", "blue"),
            ("circle", "yellow"),
        ],
    ]
    idx = 0
    for scene_i, scene in enumerate(scenes):
        img_name = f"images/scene_{scene_i:02d}.png"
        make_object_image(out / img_name, scene)
        for shape in SHAPES:
            count = sum(1 for s, _ in scene if s == shape)
            rows.append(
                {
                    "id": f"spatial_count_{idx:02d}",
                    "question": f"How many {shape}s are in this image? Reply with a number only.",
                    "answer": str(count),
                    "image": img_name,
                }
            )
            idx += 1

    with open(out / "validation.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    gen_comp_hardneg()
    gen_hallu_fg()
    gen_calib_deflect()
    gen_refcoco_mini()
    gen_spatial_count()
    print("done")
