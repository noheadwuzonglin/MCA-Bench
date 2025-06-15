import os
import json
import re

BASE_DIR = " "
OUTPUT_FILE = "data_upside_down_click.json"
VALID_IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]
VALID_TEXT_EXT = ".txt"

COORD_PATTERN = re.compile(r'^\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\);?$')


def find_files(directory, valid_exts):
    valid_exts = {ext.lower() for ext in valid_exts}
    return sorted([
        f for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in valid_exts
    ])


def parse_coords(txt_path):
    coords = []
    with open(txt_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
        for line in f:
            line = line.split('#', 1)[0].strip()
            if not line or line.lower().startswith(("pick out", "results")):
                continue
            match = COORD_PATTERN.match(line)
            if match:
                x1, y1, x2, y2 = map(int, match.groups())
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                coords.append((x1, y1, x2, y2))

    return coords if coords else None


def generate_data():
    data = []
    base_abs = os.path.abspath(BASE_DIR)
    idx = 1

    for root, _, _ in os.walk(base_abs):
        images = find_files(root, VALID_IMAGE_EXTS)
        texts = find_files(root, [VALID_TEXT_EXT])

        for img_name, txt_name in zip(images, texts):
            img_path = os.path.join(root, img_name)
            txt_path = os.path.join(root, txt_name)

            coords = parse_coords(txt_path)
            if not coords:
                continue

            rel_dir = os.path.relpath(root, base_abs).replace("\\", "/")
            rel_img = os.path.join(BASE_DIR, rel_dir, img_name).replace("\\", "/")

            clicks = []
            regions = []

            for (x1, y1, x2, y2) in coords:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                clicks.append(f"({cx},{cy})")
                regions.append([x1, y1, x2, y2])

            conversation = [
                {
                    "from": "user",
                    "value": f"Pick out all the upside down letters in the picture. <tool_response>{rel_img}</tool_response>"
                },
                {
                    "from": "assistant",
                    "value": "; ".join(clicks)
                }
            ]

            data.append({
                "id": f"captcha_{idx:06d}",
                "image_path": rel_img,
                "conversations": conversation,
                "regions": regions
            })
            idx += 1

    return data


if __name__ == "__main__":
    records = generate_data()
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

