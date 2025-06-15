import os, json, torch, pandas as pd, swanlab
from transformers import AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel, LoraConfig, TaskType
from qwen_vl_utils import process_vision_info

# ========= 路径与参数 =========
MODEL_DIR  = "../autodl-tmp/models/Qwen2.5-VL-7B-Instruct/"
LORA_DIR   = " "      # 训练脚本保存的 LoRA 目录
TEST_JSON  = "light_test.json"
CSV_OUT    = "predictionslight.csv"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


def is_in_bbox(x, y, box):

    x1,y1,x2,y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

def validate(pred_pts, regions):

    if len(pred_pts) != len(regions):
        return False
    return all(any(is_in_bbox(px,py,rb) for rb in regions) for px,py in pred_pts)

def str2pts(coord_str):

    pts = []
    for seg in coord_str.split(';'):
        seg = seg.strip()
        if not seg:
            continue
        x,y = seg.strip('() ').split(',')
        pts.append((int(x), int(y)))
    return pts


print("🔧 Loading model …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
processor = AutoProcessor.from_pretrained(MODEL_DIR)
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_DIR, device_map="auto", torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32
)

lora_cfg = LoraConfig(
    task_type      = TaskType.CAUSAL_LM,
    target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    r              = 128, lora_alpha = 16, lora_dropout = 0.0,
    inference_mode = True, bias = "none",
)
model = PeftModel.from_pretrained(base_model, LORA_DIR, config=lora_cfg).to(DEVICE).eval()

@torch.no_grad()
def predict(msgs):

    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info(msgs)
    inputs = processor(text=[text], images=img_in, videos=vid_in,
                       padding=True, return_tensors="pt").to(DEVICE)

    gen_ids = model.generate(**inputs, max_new_tokens=128)
    trimmed = [g[len(i):] for i,g in zip(inputs["input_ids"], gen_ids)]
    out_txt = processor.batch_decode(trimmed, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)[0]
    return out_txt.strip()


with open(TEST_JSON, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

records, viz_imgs, correct = [], [], 0
for ex in test_data:
    img_path = ex["image_path"]
    prompt   = ex["conversations"][0]["value"].split("<tool_response>")[0].strip()
    msgs = [{"role":"user","content":[
        {"type":"image","image":img_path},
        {"type":"text","text":prompt}
    ]}]

    pred_str = predict(msgs)
    pred_pts = str2pts(pred_str)
    passed   = validate(pred_pts, ex["regions"])
    correct += int(passed)


    records.append({
        "image":      img_path,
        "prompt":     prompt,
        "prediction": pred_str,
        "regions":    ex["regions"],
        "passed":     passed
    })

    gt_str = "; ".join(f"({int((b[0]+b[2])/2)},{int((b[1]+b[3])/2)})" for b in ex["regions"])
    viz_imgs.append(
        swanlab.Image(img_path, caption=f"GT: {gt_str}\nPR: {pred_str}\n✔️ {passed}")
    )

acc = correct / len(test_data)
print(f"\n✅ pass rate: {acc*100:.2f}% ({correct}/{len(test_data)})")
pd.DataFrame(records).to_csv(CSV_OUT, index=False)

swanlab.init(project="fret_captcha", job_type="test")
swanlab.log({
    "accuracy": acc,
    "examples": viz_imgs
})
swanlab.finish()
