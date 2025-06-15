import os
import json
import random
import torch
from PIL import Image
from datasets import Dataset
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    EarlyStoppingCallback
)
from swanlab.integration.transformers import SwanLabCallback
import swanlab


MODEL_DIR      = "../autodl-tmp/models/Qwen2.5-VL-7B-Instruct/"
DATA_JSON      = "concat.json"
TRAIN_JSON     = "concat_train.json"
VAL_JSON       = "concat_val.json"
TEST_JSON      = "concat_test.json"
OUTPUT_DIR     = "../autodl-tmp/captcha_concat_model"
MAX_LENGTH     = 8192
SEED           = 42
NUM_VAL_TEST   = 200


tokenizer  = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
processor  = AutoProcessor.from_pretrained(MODEL_DIR)

def process_func(example, max_length: int = MAX_LENGTH):

    convo          = example["conversations"]
    user_content   = convo[0]["value"]
    assistant_text = convo[1]["value"]  # 7 项指标

    try:
        img_path = user_content.split("<tool_response>")[1].split("</tool_response>")[0].strip()
        if not os.path.exists(img_path):
            raise FileNotFoundError(img_path)
    except Exception as e:
        print(e)
        return {}

    prompt_text = user_content.split("<tool_response>")[0].strip()


    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path, "resized_height": 224, "resized_width": 224},
                {"type": "text",  "text": prompt_text},
            ],
        }
    ]


    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )


    if inputs.get("image_grid_thw") is None:
        inputs["image_grid_thw"] = torch.tensor([[1, 224, 224]])


    m_inputs = {k: v.tolist() for k, v in inputs.items()}


    resp = tokenizer(assistant_text, add_special_tokens=False)

    input_ids = m_inputs["input_ids"][0] + resp["input_ids"] + [tokenizer.pad_token_id]
    att_mask  = m_inputs["attention_mask"][0] + resp["attention_mask"] + [1]
    labels    = [-100] * len(m_inputs["input_ids"][0]) + resp["input_ids"] + [tokenizer.pad_token_id]

    input_ids, att_mask, labels = [seq[:max_length] for seq in (input_ids, att_mask, labels)]

    return {
        "input_ids":      torch.tensor(input_ids),
        "attention_mask": torch.tensor(att_mask),
        "labels":         torch.tensor(labels),
        "pixel_values":   torch.tensor(m_inputs["pixel_values"]),
        "image_grid_thw": torch.tensor(m_inputs["image_grid_thw"]).squeeze(0),
    }


with open(DATA_JSON, "r", encoding="utf-8") as f:
    raw_data = json.load(f)


random.seed(SEED)
random.shuffle(raw_data)


val_test_raw = raw_data[:NUM_VAL_TEST]
train_raw = raw_data[NUM_VAL_TEST:]


json.dump(train_raw, open(TRAIN_JSON, "w", encoding="utf-8"), ensure_ascii=False)
json.dump(val_test_raw, open(VAL_JSON,   "w", encoding="utf-8"), ensure_ascii=False)
json.dump(val_test_raw, open(TEST_JSON,  "w", encoding="utf-8"), ensure_ascii=False)


train_ds = Dataset.from_json(TRAIN_JSON).map(process_func)
val_ds   = Dataset.from_json(VAL_JSON).map(process_func)


base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_DIR, device_map="auto", torch_dtype=torch.float16
)
base_model.enable_input_require_grads()

lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    inference_mode=False,
    r=64,
    lora_alpha=8,
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(base_model, lora_cfg)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    logging_steps=10,
    num_train_epochs=100,
    save_steps=100,
    eval_steps=100,
    learning_rate=1e-5,
    save_on_each_node=True,
    gradient_checkpointing=True,
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    seed=SEED,
    data_seed=SEED,
)

swanlab_cb = SwanLabCallback(
    project="concat_task",
    experiment_name="Qwen2.5VL-7B-Instruct-LoRA",
)
early_stop_cb = EarlyStoppingCallback(early_stopping_patience=20)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_cb, early_stop_cb],
)

trainer.train()

def predict(messages, mdl):
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    img_inputs, vid_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_prompt],
        images=img_inputs,
        videos=vid_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    gen_ids = mdl.generate(**inputs, max_new_tokens=128)
    gen_trim = [o[len(i):] for i, o in zip(inputs.input_ids, gen_ids)]
    out_txt  = processor.batch_decode(gen_trim, skip_special_tokens=True)
    return out_txt[0]


best_ckpt = trainer.state.best_model_checkpoint
infer_model = PeftModel.from_pretrained(base_model, best_ckpt, config=lora_cfg).to("cuda")


if __name__ == "__main__":
    sample_item = val_test_raw[0]
    uc = sample_item["conversations"][0]["value"]
    img_path = uc.split("<tool_response>")[1].split("</tool_response>")[0].strip()
    prompt_txt = uc.split("<tool_response>")[0].strip()
    msg = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img_path},
            {"type": "text",  "text": prompt_txt},
        ],
    }]
    print(predict(msg, infer_model))

    test_image_list = []
    response = predict(msg, infer_model)
    test_image_list.append(swanlab.Image(img_path, caption=response))
    swanlab.log({"Prediction": test_image_list})
    swanlab.finish()
