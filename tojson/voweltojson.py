import os
import json
import re
from pathlib import Path


BASE_DIR    = " "
OUTPUT_FILE = "vowel_data.json"
# =================================

FIXED_PROMPT = "Please click on all the vowel letters in the picture."

VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VALID_TEXT_EXT   = ".txt"

COORD_RE = re.compile(
    r"\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)",
    re.VERBOSE | re.IGNORECASE,
)


def is_image(fname: str) -> bool:

    return Path(fname).suffix.lower() in VALID_IMAGE_EXTS


def find_pairs(folder: Path):

    imgs = sorted([p for p in folder.iterdir() if is_image(p.name)])
    txts = sorted([p for p in folder.iterdir() if p.suffix.lower() == VALID_TEXT_EXT])
    return list(zip(imgs, txts))


def parse_txt(txt_path: Path):

    boxes = []

    with txt_path.open("r", encoding="utf-8-sig", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue


            m = COORD_RE.search(line)
            if m:
                x1, y1, x2, y2 = map(int, m.groups())
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                boxes.append([x1, y1, x2, y2])

    return FIXED_PROMPT, boxes


def build_samples() -> list[dict]:

    base     = Path(BASE_DIR).expanduser().resolve()
    samples  = []
    idx      = 1

    for folder, _, _ in os.walk(base):
        folder = Path(folder)
        for img_path, txt_path in find_pairs(folder):
            prompt, regions = parse_txt(txt_path)
            if not regions:
                continue

            rel_img   = img_path.relative_to(base).as_posix()
            img_token = f"{BASE_DIR}/{rel_img}"

            centers = [
                f"({(x1 + x2) // 2},{(y1 + y2) // 2})"
                for x1, y1, x2, y2 in regions
            ]
            assistant_val = "; ".join(centers)

            samples.append(
                {
                    "id"          : f"sample_{idx:06d}",
                    "image_path"  : img_token,
                    "conversations": [
                        {
                            "from" : "user",
                            "value": f"{prompt} <tool_response>{img_token}</tool_response>",
                        },
                        {
                            "from" : "assistant",
                            "value": assistant_val,
                        },
                    ],
                    "regions": regions,
                }
            )
            idx += 1

    return samples


def main():
    data = build_samples()
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
