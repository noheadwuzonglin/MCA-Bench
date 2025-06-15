from pathlib import Path
import os, json, re


BASE_DIR          = "../autodl-tmp/扭曲单词"
OUTPUT_FILE       = "word.json"
INJECT_TOOL_TAGS  = True
# ===========================

VALID_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
TXT_EXT        = ".txt"
ASCII_RE       = re.compile(r"^[\x00-\x7F]+$")


def is_image(path: Path) -> bool:
    return path.suffix.lower() in VALID_IMG_EXTS


def find_pairs(dir_path: Path):

    imgs = sorted([p for p in dir_path.iterdir() if is_image(p)])
    txts = sorted([p for p in dir_path.iterdir() if p.suffix.lower() == TXT_EXT])
    return list(zip(imgs, txts))


def parse_txt(txt_path: Path):

    prompt, answer_lines = None, []
    with txt_path.open("r", encoding="utf-8-sig", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    for idx, line in enumerate(lines):
        if ASCII_RE.fullmatch(line) and re.search(r"[A-Za-z]", line):
            prompt = line
            answer_candidates = lines[idx + 1 :]
            break

    for ln in answer_candidates:
        if ASCII_RE.fullmatch(ln) and "please" not in ln.lower():
            answer_lines.append(ln)

    answer = " ".join(answer_lines).strip()
    if not prompt or not answer:
        return None, None
    return prompt, answer


def build_samples():
    base = Path(BASE_DIR).expanduser().resolve()
    samples, idx = [], 1

    for dir_path, _, _ in os.walk(base):
        dir_path = Path(dir_path)
        for img_path, txt_path in find_pairs(dir_path):
            prompt, answer = parse_txt(txt_path)
            if not prompt:
                continue

            rel_img = img_path.relative_to(base).as_posix()
            img_token = f"{BASE_DIR}/{rel_img}"

            if INJECT_TOOL_TAGS:
                user_val = f"{prompt} <tool_response>{img_token}</tool_response>"
            else:
                user_val = prompt

            samples.append(
                {
                    "id": f"sample_{idx:06d}",
                    "image_path": img_token,
                    "conversations": [
                        {"from": "user", "value": user_val},
                        {"from": "assistant", "value": answer},
                    ],
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
