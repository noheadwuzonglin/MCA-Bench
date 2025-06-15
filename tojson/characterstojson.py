import os
import json
from glob import glob


IMAGE_DIR = r" "

# 支持的图片格式（包含大小写）
IMAGE_EXTENSIONS = [
    "*.png", "*.PNG",
    "*.jpg", "*.JPG",
    "*.jpeg", "*.JPEG",
    "*.bmp", "*.BMP",
    "*.webp", "*.WEBP"
]

image_paths = []
for ext in IMAGE_EXTENSIONS:
    image_paths.extend(glob(os.path.join(IMAGE_DIR, ext)))


conversations = []
for idx, img_path in enumerate(image_paths):

    label = os.path.splitext(os.path.basename(img_path))[0]


    conversations.append({
        "id": f"captcha_{idx + 1}",
        "conversations": [
            {
                "from": "user",
                "value": f"Please enter the characters you see on the picture: <tool_response>{img_path}</tool_call>"
            },
            {
                "from": "assistant",
                "value": label.upper()
            }
        ]
    })

# 保存为JSON文件
with open('data_vl_character.json', 'w', encoding='utf-8') as f:
    json.dump(conversations, f, ensure_ascii=False, indent=2)