import os
import json
import random
import torch
import swanlab
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from swanlab.integration.transformers import SwanLabCallback
from qwen_vl_utils import process_vision_info

BASE_JSON   = "data_upside_down_click.json"
TRAIN_JSON  = "ud_train.json"
VAL_JSON    = "ud_val.json"
TEST_JSON   = "ud_test.json"
OUTPUT_DIR  = "../autodl-tmp/ud_captcha_lora"
SEED        = 42
MAX_LENGTH  = 8192

# ============================

# ============================
def process_func(example):
    user_val      = example["conversations"][0]["value"]
    assistant_val = example["conversations"][1]["value"]


    file_path   = user_val.split("<tool_response>")[1].split("</tool_response>")[0]
    prompt_text = user_val.split("<tool_response>")[0].strip()


    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": file_path, "resized_height":224, "resized_width":224},
            {"type": "text",  "text": prompt_text}
        ]
    }]


    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )


    if inputs.get("image_grid_thw", None) is None:
        image_grid_thw = torch.tensor([1,224,224])
    else:
        image_grid_thw = inputs["image_grid_thw"].squeeze(0)


    plain = {k: v.tolist() for k, v in inputs.items()}


    resp = tokenizer(assistant_val, add_special_tokens=False)


    input_ids      = plain["input_ids"][0] + resp["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = plain["attention_mask"][0] + resp["attention_mask"] + [1]
    labels         = [-100]*len(plain["input_ids"][0]) + resp["input_ids"] + [tokenizer.pad_token_id]


    if len(input_ids) > MAX_LENGTH:
        input_ids      = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels         = labels[:MAX_LENGTH]

    return {
        "input_ids":      torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels":         torch.tensor(labels),
        "pixel_values":   torch.tensor(plain["pixel_values"]),
        "image_grid_thw": image_grid_thw
    }

# ============================

# ============================
def predict(messages, model):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info(messages)
    inputs = processor(
        text=[text], images=img_in, videos=vid_in,
        padding=True, return_tensors="pt"
    ).to("cuda")
    gen_ids = model.generate(**inputs, max_new_tokens=128)
    trimmed = [g[len(i):] for i, g in zip(inputs.input_ids, gen_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]

# ============================

# ============================
tokenizer = AutoTokenizer.from_pretrained(
    "../autodl-tmp/models/Qwen2.5-VL-7B-Instruct/", use_fast=False
)
processor = AutoProcessor.from_pretrained(
    "../autodl-tmp/models/Qwen2.5-VL-7B-Instruct/"
)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "../autodl-tmp/models/Qwen2.5-VL-7B-Instruct/",
    device_map="auto", torch_dtype=torch.float16
)
model.enable_input_require_grads()

# ============================

# ============================
with open(BASE_JSON, 'r', encoding='utf-8') as f:
    data = json.load(f)

random.seed(SEED)
random.shuffle(data)


train_data    = data[:-200]
val_test_data = data[-200:]


with open(TRAIN_JSON,  "w", encoding='utf-8') as f: json.dump(train_data,    f, ensure_ascii=False, indent=2)
with open(VAL_JSON,    "w", encoding='utf-8') as f: json.dump(val_test_data, f, ensure_ascii=False, indent=2)
with open(TEST_JSON,   "w", encoding='utf-8') as f: json.dump(val_test_data, f, ensure_ascii=False, indent=2)


train_ds = Dataset.from_json(TRAIN_JSON).map(process_func, remove_columns=None)
val_ds   = Dataset.from_json(VAL_JSON).map(process_func, remove_columns=None)

# ============================

# ============================
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    inference_mode=False,
    r=128, lora_alpha=16, lora_dropout=0.05, bias="none"
)
peft_model = get_peft_model(model, lora_config)

# ============================

# ============================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=100,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=1e-4,
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    gradient_checkpointing=True,
    seed=SEED,
    fp16=True
)

# ============================

swanlab_cb = SwanLabCallback(
    project="ud_captcha",
    experiment_name="Qwen2.5VL-UD",
    config={"model":"qwen/Qwen2-1.5B-Instruct","dataset":"UpsideDownLetters"}
)
early_stop = EarlyStoppingCallback(early_stopping_patience=35)


trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_cb, early_stop]
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_peft"))

    best_ckpt = trainer.state.best_model_checkpoint
    val_peft = PeftModel.from_pretrained(model, best_ckpt, config=lora_config).to("cuda")

    with open(TEST_JSON, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    test_image_list = []
    for item in test_data:
        inp = item["conversations"][0]["value"]
        img = inp.split("<tool_response>")[1].split("</tool_response>")[0]
        prompt = inp.split("<tool_response>")[0].strip()
        msgs = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":prompt}]}]
        resp = predict(msgs, val_peft)
        test_image_list.append(swanlab.Image(img, caption=resp))

    swanlab.log({"Prediction": test_image_list})
    swanlab.finish()
