import json
import re
from pathlib import Path
import os

BASE_DIR    = " "
OUTPUT_FILE = "light_data.json"

VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VALID_TEXT_EXT   = ".txt"


COORD_RE = re.compile(
    r"\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)",
    re.VERBOSE | re.IGNORECASE,
)


IS_ASCII_RE = re.compile(r"^[\x00-\x7F]+$")


def is_image(fname: str) -> bool:
    return Path(fname).suffix.lower() in VALID_IMAGE_EXTS


def find_pairs(folder: Path):
    imgs = sorted([p for p in folder.iterdir() if is_image(p.name)])
    txts = sorted([p for p in folder.iterdir() if p.suffix.lower() == VALID_TEXT_EXT])
    return list(zip(imgs, txts))


def parse_txt(txt_path: Path):
    prompt = None
    boxes  = []

    with txt_path.open("r", encoding="utf-8-sig", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue


            if prompt is None and IS_ASCII_RE.fullmatch(line) and re.search(r"[A-Za-z]", line):
                prompt = line
                continue


            m = COORD_RE.search(line)
            if m:
                x1, y1, x2, y2 = map(int, m.groups())
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                boxes.append([x1, y1, x2, y2])

    return prompt, boxes


def build_samples() -> list[dict]:

    base     = Path(BASE_DIR).expanduser().resolve()
    samples  = []
    idx      = 1

    for folder, _, _ in os.walk(base):
        folder = Path(folder)
        for img_path, txt_path in find_pairs(folder):
            prompt, regions = parse_txt(txt_path)
            if not prompt or not regions:
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
                    "regions": regions,   # [[x1,y1,x2,y2], ...]
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
