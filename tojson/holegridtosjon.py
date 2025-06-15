import os
import json


BASE_DIR = " "
OUTPUT_FILE = "data_vl_holegrid.json"
VALID_IMAGE_EXTS = [".png", ".jpg", ".jpeg"]


def find_image_and_txt(directory):
    image_file = None
    txt_file = None
    for fname in os.listdir(directory):
        ext = os.path.splitext(fname)[1].lower()
        if ext in VALID_IMAGE_EXTS:
            image_file = fname
        elif ext == ".txt":
            txt_file = fname
    return image_file, txt_file


def parse_meta(meta_path):
    with open(meta_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if len(lines) < 3:
        return {"verification": None, "click_indices": None}

    verification = lines[1]
    click_indices = lines[2]

    return {
        "verification": verification,
        "click_indices": click_indices
    }


def generate_conversations():
    conversations = []
    base_abs_path = os.path.abspath(BASE_DIR)

    for root, _, _ in os.walk(base_abs_path):
        if root == base_abs_path:
            continue

        image_file, txt_file = find_image_and_txt(root)
        if not image_file or not txt_file:
            continue

        meta_path = os.path.join(root, txt_file)
        meta = parse_meta(meta_path)
        if not meta["verification"] or not meta["click_indices"]:
            continue

        relative_path = os.path.relpath(root, base_abs_path).replace("\\", "/")

        image_path = os.path.join(BASE_DIR, relative_path, image_file).replace("\\", "/")

        conversations.append({
            "id": f"captcha_{len(conversations) + 1:04d}",
            "conversations": [
                {
                    "from": "user",
                    "value": f"{meta['verification']} <tool_response>{image_path}</tool_call>"
                },
                {
                    "from": "assistant",
                    "value": meta["click_indices"]
                }
            ]
        })

    return conversations


if __name__ == "__main__":
    data = generate_conversations()
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

