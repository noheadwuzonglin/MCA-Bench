import os
import json
import re

BASE_DIR = " "
OUTPUT_FILE = "data_vl_click_c.json"
VALID_IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]
VALID_TEXT_EXT = ".txt"

COORD_PATTERN = re.compile(r'''
    ^\s*                    
    ([A-Za-z])            
    \s*                  
    [:(（]                 
    \s*                    
    (\d+)\s*,\s*(\d+)      
    \s*,\s*               
    (\d+)\s*,\s*(\d+)      
    \s*                     
    [):）]                
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
        line_clean = line.split('#', 1)[0].strip()  # 去除注释和空格
        if not line_clean:
            continue


        if COORD_PATTERN.match(line_clean):
            coordinates.append(line_clean)
        else:

            if re.match(r'^Please click on [A-Z]+ in turn\.?$', line_clean, re.IGNORECASE):
                instructions.append(line_clean)



    verification = instructions[0].strip() if instructions else ""

    click_sequence = []
    regions = {}
    for line in coordinates:
        match = COORD_PATTERN.match(line)

        letter = match.group(1).upper()
        x1 = int(match.group(2))
        y1 = int(match.group(3))
        x2 = int(match.group(4))
        y2 = int(match.group(5))


        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        click_sequence.append(f"{letter} ({cx},{cy})")
        regions[letter] = (x1, y1, x2, y2)

    return {
        "verification": verification,
        "click_sequence": click_sequence,
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
                        "value": " -> ".join(meta["click_sequence"])
                    }
                ],
                "regions": meta["regions"]
            })

    return conversations


if __name__ == "__main__":
    data = generate_conversations()
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

