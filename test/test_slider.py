import os, json, re
from typing import Dict, List

import torch
from PIL import Image
from transformers import (
    AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration,
)
from peft import PeftModel, LoraConfig, TaskType


MODEL_DIR = "../autodl-tmp/models/Qwen2.5-VL-7B-Instruct/"
PEFT_DIR  = " "
BASE_JSON = "slider.json"
TEST_JSON = "slider_test.json"

IMG_SIZE  = (224, 224)
SIGMA         = 1
DIST_TOL      = 0.20

METRIC_RE = re.compile(
    r"duration=([-+]?\d*\.?\d+),\s*"
    r"distance=([-+]?\d*\.?\d+),\s*"
    r"speed_min=([-+]?\d*\.?\d+),\s*"
    r"speed_max=([-+]?\d*\.?\d+),\s*"
    r"acc_min=([-+]?\d*\.?\d+),\s*"
    r"acc_max=([-+]?\d*\.?\d+),\s*"
    r"jitter_avg=([-+]?\d*\.?\d+)"
)


METRIC_NAMES = [
    "duration", "distance",
    "speed_min", "speed_max",
    "acc_min", "acc_max", "jitter_avg"
]

METRIC_NAMES_NO_DIST = [
    "duration",
    "speed_min", "speed_max",
    "acc_min", "acc_max", "jitter_avg"
]

# -------------------------------------------------

# -------------------------------------------------
def extract_images_from_user_value(text: str) -> List[str]:
    out, start = [], 0
    while True:
        start = text.find("<tool_response>", start)
        if start == -1: break
        end = text.find("</tool_response>", start)
        if end == -1: break
        out.append(text[start + 15:end].strip())
        start = end + 16
    return out


def concat_images(paths: List[str], size=(224, 224)):
    from PIL import Image
    assert len(paths) == 2
    imgs = [Image.open(p).convert("RGB").resize(size) for p in paths]
    w, h = size
    canvas = Image.new("RGB", (2*w, h))
    canvas.paste(imgs[0], (0, 0));  canvas.paste(imgs[1], (w, 0))
    return canvas


def build_prompt() -> str:
    return "Drag the slider to complete the puzzle."


def parse_metrics(txt: str) -> Dict[str, float]:
    m = METRIC_RE.search(txt)

    return {k: float(v) for k, v in zip(METRIC_NAMES, m.groups())}


def compute_dataset_stats(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    vals = {k: [] for k in METRIC_NAMES_NO_DIST}
    for ex in data:
        m = parse_metrics(ex["conversations"][1]["value"])
        for k in METRIC_NAMES_NO_DIST:
            vals[k].append(m[k])
    stats = {}
    for k in METRIC_NAMES_NO_DIST:
        t = torch.tensor(vals[k], dtype=torch.float32)
        stats[k] = {"mean": t.mean().item(), "std": t.std().item()}
    return stats


# -------------------------------------------------
# -------------------------------------------------
def predict(img_paths, model, processor):
    comp = concat_images(img_paths, size=IMG_SIZE)
    msgs = [{
        "role": "user",
        "content": [
            {"type": "image", "image": comp,
             "resized_height": comp.height, "resized_width": comp.width},
            {"type": "text", "text": build_prompt()}
        ]
    }]
    prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    device = next(model.parameters()).device
    inputs = processor(text=[prompt], images=[comp],
                       padding=True, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=64)
    gen_only = out[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(gen_only, skip_special_tokens=True)[0]


# -------------------------------------------------

# -------------------------------------------------
def evaluate(test_json, model, processor, stats):
    with open(test_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    total, passed = 0, 0
    for ex in data:
        img_paths = extract_images_from_user_value(ex["conversations"][0]["value"])
        pred_txt  = predict(img_paths, model, processor).strip()
        true_m    = parse_metrics(ex["conversations"][1]["value"])
        pred_m    = parse_metrics(pred_txt)

        ok = True


        rel = abs(pred_m["distance"] - true_m["distance"]) / max(true_m["distance"], 1e-6)
        if rel > DIST_TOL:
            ok = False


        for k in METRIC_NAMES_NO_DIST:
            mu, sigma = stats[k]['mean'], stats[k]['std']
            if sigma == 0:
                if abs(pred_m[k] - mu) > 1e-6:
                    ok = False
            else:
                if not (mu - SIGMA*sigma <= pred_m[k] <= mu + SIGMA*sigma):
                    ok = False


        print("="*60)
        print(f'ID: {ex.get("id", "N/A")}')
        print("Pred :", pred_txt)
        print("Truth:", ex["conversations"][1]["value"].strip())
        print("Pass :", ok)

        total += 1
        passed += int(ok)

    acc = passed/total if total else 0
    print(f"\nOverall Pass-Rate: {acc*100:.2f}%  ({passed}/{total})")
    return acc


# -------------------------------------------------

# -------------------------------------------------
def main():

    os.environ["CUDNN_V8_API_ENABLED"] = "0"
    torch.backends.cuda.matmul.allow_tf32 = False

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    base      = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_DIR, device_map="auto", torch_dtype=torch.float16
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        r=64, lora_alpha=8, lora_dropout=0.05, bias="none", inference_mode=False
    )
    model = PeftModel.from_pretrained(base, PEFT_DIR, config=lora_cfg).eval()

    stats = compute_dataset_stats(BASE_JSON)
    for k in METRIC_NAMES_NO_DIST:
        print(f"{k:10}: {stats[k]['mean']:.4f} ± {stats[k]['std']:.4f}")

    evaluate(TEST_JSON, model, processor, stats)


if __name__ == "__main__":
    main()
