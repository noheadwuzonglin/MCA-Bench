import os, json, random, torch, swanlab
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments, Trainer,
    DataCollatorForSeq2Seq, EarlyStoppingCallback,
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from swanlab.integration.transformers import SwanLabCallback
from qwen_vl_utils import process_vision_info


BASE_JSON   = "light_data.json"
TRAIN_JSON  = "light_train.json"
VAL_JSON    = "light_val.json"
TEST_JSON   = "light_test.json"
OUTPUT_DIR  = "../autodl-tmp/light"
SEED        = 42
MAX_LENGTH  = 8192


def bbox2center(b):

    cx = int(round((b[0] + b[2]) / 2))
    cy = int(round((b[1] + b[3]) / 2))
    return f"({cx},{cy})"

def regions_to_answer(regions):

    return "; ".join(bbox2center(b) for b in regions)


def process_func(example):
    # ---------- prompt ----------
    user_raw = example["conversations"][0]["value"]
    prompt   = user_raw.split("<tool_response>")[0].strip()


    img_path = example["image_path"]
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img_path, "resized_height": 224, "resized_width": 224},
            {"type": "text",  "text": prompt},
        ]
    }]


    if len(example["conversations"]) > 1 and example["conversations"][1]["value"].strip():
        assistant_val = example["conversations"][1]["value"].strip()
    else:
        assistant_val = regions_to_answer(example["regions"])


    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=img_in,
        videos=vid_in,
        padding=True,
        return_tensors="pt"
    )


    image_grid_thw = inputs.get("image_grid_thw", torch.tensor([[1,224,224]])).squeeze(0)

    plain = {k: v.tolist() for k, v in inputs.items()}
    resp  = tokenizer(assistant_val, add_special_tokens=False)

    input_ids      = plain["input_ids"][0] + resp["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = plain["attention_mask"][0] + resp["attention_mask"] + [1]
    labels         = [-100]*len(plain["input_ids"][0]) + resp["input_ids"] + [tokenizer.pad_token_id]


    if len(input_ids) > MAX_LENGTH:
        input_ids, attention_mask, labels = (
            input_ids[:MAX_LENGTH],
            attention_mask[:MAX_LENGTH],
            labels[:MAX_LENGTH]
        )

    return {
        "input_ids":      torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels":         torch.tensor(labels),
        "pixel_values":   torch.tensor(plain["pixel_values"]),
        "image_grid_thw": image_grid_thw,
    }


tokenizer  = AutoTokenizer.from_pretrained("../autodl-tmp/models/Qwen2.5-VL-7B-Instruct/", use_fast=False)
processor  = AutoProcessor.from_pretrained("../autodl-tmp/models/Qwen2.5-VL-7B-Instruct/")
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "../autodl-tmp/models/Qwen2.5-VL-7B-Instruct/",
    device_map="auto", torch_dtype=torch.float16
)
base_model.enable_input_require_grads()


with open(BASE_JSON, 'r', encoding='utf-8') as f:
    data = json.load(f)

random.seed(SEED); random.shuffle(data)
train_data, val_test_data = data[:-200], data[-200:]

for fn, subset in zip((TRAIN_JSON, VAL_JSON, TEST_JSON),
                      (train_data, val_test_data, val_test_data)):
    with open(fn, "w", encoding="utf-8") as f: json.dump(subset, f, ensure_ascii=False, indent=2)

train_ds = Dataset.from_json(TRAIN_JSON).map(process_func, remove_columns=None)
val_ds   = Dataset.from_json(VAL_JSON).map(process_func, remove_columns=None)


lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    r=128, lora_alpha=16, lora_dropout=0.05, bias="none",
    inference_mode=False,
)
model = get_peft_model(base_model, lora_cfg)


training_args = TrainingArguments(
    output_dir              = OUTPUT_DIR,
    per_device_train_batch_size = 8,
    gradient_accumulation_steps = 4,
    num_train_epochs        = 100,
    save_steps              = 100,
    eval_steps              = 100,
    logging_steps           = 10,
    learning_rate           = 1e-4,
    evaluation_strategy     = "steps",
    save_strategy           = "steps",
    load_best_model_at_end  = True,
    metric_for_best_model   = "eval_loss",
    greater_is_better       = False,
    gradient_checkpointing  = True,
    seed                    = SEED,
    fp16                    = True,
)


swanlab_cb = SwanLabCallback(
    project="fretwork_captcha",
    experiment_name="Qwen2.5VL-LoRA",
    config={"model":"Qwen2.5VL-7B-Instruct","notes":"click-all-fretwork"}
)
early_stop = EarlyStoppingCallback(early_stopping_patience=35)

# ========= Trainer =========
trainer = Trainer(
    model         = model,
    args          = training_args,
    train_dataset = train_ds,
    eval_dataset  = val_ds,
    data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True),
    callbacks     = [swanlab_cb, early_stop],
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_peft"))


    best_ckpt = trainer.state.best_model_checkpoint
    infer_model = PeftModel.from_pretrained(base_model, best_ckpt, config=lora_cfg).to("cuda")

    def predict(msgs):
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        img_in, vid_in = process_vision_info(msgs)
        inputs = processor(text=[text], images=img_in, videos=vid_in,
                           padding=True, return_tensors="pt").to("cuda")
        gen = infer_model.generate(**inputs, max_new_tokens=128)
        trimmed = [g[len(i):] for i, g in zip(inputs.input_ids, gen)]
        return processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    with open(TEST_JSON, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    viz = []
    for item in test_data:
        img  = item["image_path"]
        text = item["conversations"][0]["value"].split("<tool_response>")[0].strip()
        msgs = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":text}]}]
        viz.append(swanlab.Image(img, caption=predict(msgs)))

    swanlab.log({"Prediction": viz})
    swanlab.finish()
