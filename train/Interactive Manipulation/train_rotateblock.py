import os
import json
import random
from typing import List

import torch
from PIL import Image
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from swanlab.integration.transformers import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model

BASE_JSON   = "rotate.json"
TRAIN_JSON  = "rotate_train.json"
VAL_JSON    = "rotate_val.json"
TEST_JSON   = "rotate_test.json"

MODEL_DIR   = "../autodl-tmp/models/Qwen2.5-VL-7B-Instruct/"
OUTPUT_DIR  = "../autodl-tmp/rotate_concat_lora"
SEED        = 42
MAX_LENGTH  = 8192
IMG_SIZE    = (224, 224)
PATCH       = 14


def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_images_from_user_value(user_value: str) -> List[str]:

    paths, start = [], 0
    while True:
        start = user_value.find("<tool_response>", start)
        if start == -1:
            break
        end = user_value.find("</tool_response>", start)
        if end == -1:
            break
        paths.append(user_value[start + len("<tool_response>"): end].strip())
        start = end + len("</tool_response>")
    return paths


def concat_images(img_paths: List[str], resize=(224, 224), mode: str = "horizontal") -> Image.Image:


    imgs = [Image.open(p).convert("RGB").resize(resize) for p in img_paths]
    w, h = resize
    if mode == "horizontal":
        composite = Image.new("RGB", (w * 2, h))
        composite.paste(imgs[0], (0, 0))
        composite.paste(imgs[1], (w, 0))
    else:
        composite = Image.new("RGB", (w, h * 2))
        composite.paste(imgs[0], (0, 0))
        composite.paste(imgs[1], (0, h))
    return composite



def build_prompt() -> str:
    return (
        "Please rotate the puzzle to the correct angle."
    )


def process_func(example, processor, tokenizer):

    conv       = example["conversations"]
    user_val   = conv[0]["value"]
    assistant  = conv[1]["value"]


    img_paths = extract_images_from_user_value(user_val)
    comp_img  = concat_images(img_paths, resize=IMG_SIZE)          # 224×448


    h_patch, w_patch = comp_img.height // PATCH, comp_img.width // PATCH
    image_grid_thw   = torch.tensor([1, h_patch, w_patch], dtype=torch.long)


    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": comp_img,
             "resized_height": comp_img.height, "resized_width": comp_img.width},
            {"type": "text",  "text": build_prompt()},
        ],
    }]
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text_prompt],
        images=[comp_img],
        padding=True,
        return_tensors="pt",
    )


    resp_tokens  = tokenizer(assistant, add_special_tokens=False)
    input_ids    = torch.cat([inputs["input_ids"][0],
                              torch.tensor(resp_tokens["input_ids"]),
                              torch.tensor([tokenizer.pad_token_id])])
    attention_ms = torch.cat([inputs["attention_mask"][0],
                              torch.tensor(resp_tokens["attention_mask"]),
                              torch.tensor([1])])
    labels       = torch.cat([torch.full_like(inputs["input_ids"][0], -100),
                              torch.tensor(resp_tokens["input_ids"]),
                              torch.tensor([tokenizer.pad_token_id])])


    if input_ids.size(0) > MAX_LENGTH:
        input_ids    = input_ids[:MAX_LENGTH]
        attention_ms = attention_ms[:MAX_LENGTH]
        labels       = labels[:MAX_LENGTH]

    return {
        "input_ids":       input_ids,
        "attention_mask":  attention_ms,
        "labels":          labels,
        "pixel_values":    inputs["pixel_values"].squeeze(0),
        "image_grid_thw":  image_grid_thw,
    }



def main():
    set_seed(SEED)

    if not (os.path.exists(TRAIN_JSON) and os.path.exists(VAL_JSON)):
        if not os.path.exists(BASE_JSON):
            raise FileNotFoundError(f"找不到原始数据文件 {BASE_JSON}")
        with open(BASE_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)

        random.seed(SEED)

        val_test_indices = random.sample(range(len(data)), 200)
        val_test_data    = [data[i] for i in val_test_indices]
        train_data       = [d for idx, d in enumerate(data) if idx not in val_test_indices]

        for fn, subset in zip((TRAIN_JSON, VAL_JSON, TEST_JSON),
                          (train_data, val_test_data, val_test_data)):
            with open(fn, "w", encoding="utf-8") as f:
                json.dump(subset, f, ensure_ascii=False, indent=2)



    tokenizer  = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
    processor  = AutoProcessor.from_pretrained(MODEL_DIR)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_DIR, device_map="auto", torch_dtype=torch.float16
    )

    base_model.enable_input_require_grads()


    lora_cfg = LoraConfig(
        task_type      = TaskType.CAUSAL_LM,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
        r              = 64,
        lora_alpha     = 8,
        lora_dropout   = 0.05,
        bias           = "none",
        inference_mode = False,
    )
    model = get_peft_model(base_model, lora_cfg)


    train_ds = Dataset.from_json(TRAIN_JSON)
    val_ds   = Dataset.from_json(VAL_JSON)

    map_fn   = lambda ex: process_func(ex, processor, tokenizer)
    train_ds = train_ds.map(map_fn, remove_columns=train_ds.column_names)
    val_ds   = val_ds.map(map_fn,   remove_columns=val_ds.column_names)


    swanlab_callback = SwanLabCallback(
        project="captcha-ocr-project",
        experiment_name="qwen2.5-vl-captcha",
        config={
            "model": "Qwen2.5-VL-7B-Instruct",
            "dataset_info": "local_captcha_data",
            "prompt_example": "Please enter the characters you see on the picture:",
            "train_data_number": len(train_ds),
            "lora_rank": lora_cfg.r,
            "lora_alpha": lora_cfg.lora_alpha,
            "lora_dropout": lora_cfg.lora_dropout,
        },
    )


    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=30,
    )


    training_args = TrainingArguments(
        output_dir                  = OUTPUT_DIR,
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 8,
        num_train_epochs            = 100,
        learning_rate               = 1e-5,
        logging_steps               = 10,
        save_steps                  = 100,
        eval_steps                  = 100,
        evaluation_strategy         = "steps",
        save_strategy               = "steps",
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        gradient_checkpointing      = True,
        fp16                        = True,
        seed                        = SEED,
        remove_unused_columns       = False,
    )


    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        data_collator   = DataCollatorForSeq2Seq(tokenizer, padding=True),
        callbacks       = [swanlab_callback, early_stopping],
    )


    trainer.train()
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_peft"))


    def build_msg(img_paths: List[str], prompt_text: str):
        comp_img = concat_images(img_paths, resize=IMG_SIZE)
        return [{
            "role": "user",
            "content": [
                {"type": "image", "image": comp_img},
                {"type": "text",  "text": prompt_text},
            ],
        }]

    def predict(msgs):
        text_prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs        = [msg["content"][0]["image"] for msg in msgs]
        inputs      = processor(text=[text_prompt], images=imgs,
                                 padding=True, return_tensors="pt").to("cuda")
        gen         = model.generate(**inputs, max_new_tokens=128)
        trimmed     = [g[len(i):] for i, g in zip(inputs.input_ids, gen)]
        return processor.batch_decode(trimmed, skip_special_tokens=True)[0]


if __name__ == "__main__":
    main()
