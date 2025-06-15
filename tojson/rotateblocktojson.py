from pathlib import Path
from collections import defaultdict
import pandas as pd, numpy as np, json


CSV_FILE   = " "
IMAGES_DIR = " "
OUTPUT_FILE = "rotate.json"
# =========================

VALID_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
TOOL_TMPL      = "<tool_response>{}</tool_response>"

def find_image(data_id: str, kw: str):

    for p in Path(IMAGES_DIR).iterdir():
        if p.suffix.lower() in VALID_IMG_EXTS and kw in p.name and str(data_id) in p.name:
            return p.resolve()

def metrics(rows):

    x   = [r.x for r in rows]
    t   = [r.t for r in rows]
    spd = [r.speed for r in rows if pd.notna(r.speed)]
    acc = [r.acceleration for r in rows if pd.notna(r.acceleration)]
    jit = [r.jitter_frequency for r in rows if pd.notna(r.jitter_frequency)]

    ms = {
        "duration":   round(((t[-1] - t[0]) / 1000) if len(t) > 1 else 0, 3),
        "distance":   round((x[-1] - x[0]) if len(x) > 1 else 0, 2),
        "speed_min":  round(float(np.min(spd)) if spd else 0, 2),
        "speed_max":  round(float(np.max(spd)) if spd else 0, 2),
        "acc_min":    round(float(np.min(acc)) if acc else 0, 2),
        "acc_max":    round(float(np.max(acc)) if acc else 0, 2),
        "jitter_avg": round(float(np.mean(jit)) if jit else 0, 2),
    }
    return ", ".join(f"{k}={v}" for k, v in ms.items())

def main():

    df = (
        pd.read_csv(CSV_FILE)
        .query("type == 'ROTATE'")
        .sort_values(["data_id", "group_index"])
        .groupby("data_id", as_index=False)
        .head(10)
    )


    groups = defaultdict(list)
    for r in df.itertuples(index=False):
        groups[r.data_id].append(r)

    samples, miss, idx = [], [], 1
    for did, rows in groups.items():
        tg = find_image(did, "ROTATE_tg")
        bg = find_image(did, "ROTATE_bg")
        if not tg or not bg:
            miss.append(did)
            continue

        user_val = (
            "Please rotate the puzzle to the correct angle.\n"
            f"{TOOL_TMPL.format(f'{IMAGES_DIR}/{tg.name}')}\n"
            f"{TOOL_TMPL.format(f'{IMAGES_DIR}/{bg.name}')}"
        )

        samples.append({
            "id": f"sample_{idx:06d}",
            "conversations": [
                {"from": "user",      "value": user_val},
                {"from": "assistant", "value": metrics(rows)},
            ],
        })
        idx += 1


    Path(OUTPUT_FILE).write_text(
        json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8"
    )

if __name__ == "__main__":
    main()
