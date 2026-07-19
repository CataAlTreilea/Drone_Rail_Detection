import json
from collections import defaultdict
from pathlib import Path

NDJSON = Path("railsv2.ndjson")
OUT_ROOT = Path(".") / "Dataset" / ""          # change to wherever you want it
(OUT_ROOT / "labels/train").mkdir(parents=True, exist_ok=True)
(OUT_ROOT / "labels/val").mkdir(parents=True, exist_ok=True)

class_names = {}
images = {}                                # file -> {W, H, split}
anns_by_image = defaultdict(list)          # file -> list of annotation records

with NDJSON.open() as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        t = rec.get("type")

        if t == "dataset":
            class_names = {int(k): v for k, v in rec["class_names"].items()}

        elif t == "image":
            images[rec["file"]] = {
                "W": rec["width"],
                "H": rec["height"],
                "split": rec.get("split", "train"),
            }

        elif t in ("annotation", "label"):       # adjust once you confirm the type
            anns_by_image[rec["file"]].append(rec)

        # silently ignore anything else

# --- write YOLO seg label files ---
for fname, meta in images.items():
    W, H, split = meta["W"], meta["H"], meta["split"]
    stem = Path(fname).stem
    lines = []
    for ann in anns_by_image.get(fname, []):
        cls = ann["class"]                       # <-- depends on actual field name
        # SEGMENTATION: polygon as flat list of pixel coords [x1,y1,x2,y2,...]
        poly = ann["polygon"]                    # <-- depends on actual field name
        norm = []
        for i, v in enumerate(poly):
            norm.append(f"{(v / W if i % 2 == 0 else v / H):.6f}")
        lines.append(f"{cls} " + " ".join(norm))
    (OUT_ROOT / f"labels/{split}/{stem}.txt").write_text("\n".join(lines))

# --- write the dataset YAML ---
yaml_text = f"""path: {OUT_ROOT}
train: images/train
val: images/val
names:
"""
for i, name in sorted(class_names.items()):
    yaml_text += f"  {i}: {name}\n"
(OUT_ROOT / "Railsv2.yaml").write_text(yaml_text)

print(f"Wrote labels for {len(images)} images, {sum(len(v) for v in anns_by_image.values())} annotations")