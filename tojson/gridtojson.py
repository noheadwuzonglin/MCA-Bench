import os
import re
import json

BASE_DIR = " "
OUTPUT_FILE = "data_vl_grid.json"
VALID_IMAGE_EXTS = [".png", ".jpg", ".jpeg"]


def find_image_file(directory):
    for fname in os.listdir(directory):
        name, ext = os.path.splitext(fname)
        if name == "grid" and ext.lower() in VALID_IMAGE_EXTS:
            return fname
    return None


def parse_meta(meta_path):
    with open(meta_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.read().splitlines()

    verification = None
    click_indices = None

    for line in lines:
        line = line.strip()
        if line.lower().startswith("verification:"):
            verification = line.split(":", 1)[-1].strip()
        elif line.startswith("right position："):
            click_indices = line.split("：", 1)[-1].strip()

    return {
        "verification": verification,
        "click_indices": click_indices
    }


def generate_conversations():
    conversations = []
    base_abs_path = os.path.abspath(BASE_DIR)

    for root, _, _ in os.walk(base_abs_path):
        relative_path = os.path.relpath(root, base_abs_path).replace("\\", "/")

        image_file = find_image_file(root)
        if not image_file:
            continue

        meta_path = os.path.join(root, "meta.txt")
        if not os.path.exists(meta_path):
            continue

        meta = parse_meta(meta_path)
        if not meta["verification"] or not meta["click_indices"]:
            continue

        if relative_path == ".":
            image_path = f"../autodl-tmp/generated_captchas/{image_file}"
        else:
            image_path = f"../autodl-tmp/generated_captchas/{relative_path}/{image_file}"

        # 统一路径分隔符
        image_path = image_path.replace("\\", "/")

        # 添加到数据集
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
