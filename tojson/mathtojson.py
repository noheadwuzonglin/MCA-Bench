import os
import json
import re


BASE_DIR = " "
OUTPUT_FILE = "data_vl_math.json"
VALID_IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]
VALID_TEXT_EXT = ".txt"


RESULT_PATTERN = re.compile(r"^results:\s*(.*)$", re.IGNORECASE)


def find_files(directory, valid_exts):
    valid_exts = [ext.lower() for ext in valid_exts]
    return [
        f for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in valid_exts
    ]


def parse_meta(meta_path):
    try:
        with open(meta_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
            lines = [line.rstrip('\n') for line in f]
    except Exception as e:
        print(f"{meta_path} ({str(e)})")
        return None


    result = None
    for line in lines:
        line_clean = line.strip()
        if RESULT_PATTERN.match(line_clean):
            result = RESULT_PATTERN.match(line_clean).group(1).strip()
            break

    return result


def generate_conversations():

    conversations = []
    base_abs_path = os.path.abspath(BASE_DIR)

    for root, _, _ in os.walk(base_abs_path):

        images = find_files(root, VALID_IMAGE_EXTS)
        texts = find_files(root, [VALID_TEXT_EXT])


        for img_file, txt_file in zip(images, texts):
            meta_path = os.path.join(root, txt_file)
            result = parse_meta(meta_path)

            if not result:
                continue


            relative_path = os.path.relpath(root, base_abs_path).replace("\\", "/")
            image_path_abs = os.path.join(base_abs_path, relative_path, img_file)
            image_path_abs = image_path_abs.replace("\\", "/")


            image_path = os.path.join(BASE_DIR, relative_path, img_file)


            conversations.append({
                "id": f"captcha_{len(conversations) + 1:06d}",
                "image_path": image_path,
                "conversations": [
                    {
                        "from": "user",
                        "value": f"Please calculate the result in the image <tool_response>{image_path}</tool_response>"
                    },
                    {
                        "from": "assistant",
                        "value": result
                    }
                ]
            })

    return conversations


if __name__ == "__main__":
    data = generate_conversations()
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

