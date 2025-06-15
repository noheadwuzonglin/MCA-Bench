import os
import json
import re


BASE_DIR = ""
OUTPUT_FILE = "data_vl_click_math.json"
VALID_IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]
VALID_TEXT_EXT = ".txt"

# 坐标格式正则表达式
COORD_PATTERN = re.compile(r'''
    ^\s*                    
    (\d+)                  
    \s*                    
    :\s*                     
    \(\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*\)  # (x1, y1, x2, y2)
    \s*                      
    (?:\#.*)?                
    $                        
''', re.IGNORECASE | re.VERBOSE)


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


    instructions = []
    coordinates = []
    for line in lines:
        line_clean = line.split('#', 1)[0].strip()
        if not line_clean:
            continue


        if re.match(r'^Please click on \w+ numbers in the picture so that their sum is \d+$', line_clean, re.IGNORECASE):
            instructions.append(line_clean)


        elif COORD_PATTERN.match(line_clean):
            coordinates.append(line_clean)

    # 只取第一个有效指令
    verification = instructions[0].strip() if instructions else ""

    # 解析坐标
    regions = {}
    for line in coordinates:
        match = COORD_PATTERN.match(line)

        number = match.group(1)
        x1 = int(match.group(2))
        y1 = int(match.group(3))
        x2 = int(match.group(4))
        y2 = int(match.group(5))

        # 自动修正坐标顺序
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        regions[number] = (x1, y1, x2, y2)

    # 返回解析结果
    return {
        "verification": verification,
        "regions": regions
    }


def generate_conversations():

    conversations = []
    base_abs_path = os.path.abspath(BASE_DIR)

    for root, _, _ in os.walk(base_abs_path):

        images = find_files(root, VALID_IMAGE_EXTS)
        texts = find_files(root, [VALID_TEXT_EXT])


        for img_file, txt_file in zip(images, texts):
            meta_path = os.path.join(root, txt_file)
            meta = parse_meta(meta_path)

            if not meta:
                continue

            relative_path = os.path.relpath(root, base_abs_path).replace("\\", "/")
            image_path_abs = os.path.join(base_abs_path, relative_path, img_file)
            image_path_abs = image_path_abs.replace("\\", "/")


            image_path = os.path.join(BASE_DIR, relative_path, img_file)

            click_sequence = ", ".join([f"{num} ({(x1 + x2) // 2},{(y1 + y2) // 2})" for num, (x1, y1, x2, y2) in meta["regions"].items()])


            conversations.append({
                "id": f"captcha_{len(conversations) + 1:06d}",
                "image_path": image_path,
                "conversations": [
                    {
                        "from": "user",
                        "value": f"{meta['verification']} <tool_response>{image_path}</tool_response>"
                    },
                    {
                        "from": "assistant",
                        "value": click_sequence
                    }
                ],
                "regions": meta["regions"]
            })

    return conversations


if __name__ == "__main__":
    data = generate_conversations()
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

