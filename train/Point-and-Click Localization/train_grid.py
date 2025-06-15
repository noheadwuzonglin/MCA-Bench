import os
import torch
import json
import swanlab
from PIL import Image
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.transformers import SwanLabCallback
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    EarlyStoppingCallback
)


# ============================

# ============================
def process_func(example):

    MAX_LENGTH = 8192

    conversation = example["conversations"]

    input_content = conversation[0]["value"]
    output_content = conversation[1]["value"]

    file_path = input_content.split("<tool_response>")[1].split("</tool_call>")[0]

    prompt_text = input_content.split("<tool_response>")[0].strip()


    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": file_path,
                    "resized_height": 224,
                    "resized_width": 224,
                },
                {
                    "type": "text",
                    "text": prompt_text
                },
            ],
        }
    ]


    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )


    inputs = {key: value.tolist() for key, value in inputs.items()}

    response = tokenizer(output_content, add_special_tokens=False)


    input_ids = (
            inputs["input_ids"][0]
            + response["input_ids"]
            + [tokenizer.pad_token_id]
    )


    attention_mask = (
            inputs["attention_mask"][0]
            + response["attention_mask"]
            + [1]
    )


    labels = (
            [-100] * len(inputs["input_ids"][0])
            + response["input_ids"]
            + [tokenizer.pad_token_id]
    )


    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]


    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    labels = torch.tensor(labels)

    pixel_values = torch.tensor(inputs["pixel_values"])
    image_grid_thw = torch.tensor(inputs["image_grid_thw"]).squeeze(0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }


# ============================

# ============================
def predict(messages, model):

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")


    generated_ids = model.generate(**inputs, max_new_tokens=128)
    # 分割 prompt 与新生成 token
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    return output_text[0]


# ============================

# ============================
tokenizer = AutoTokenizer.from_pretrained(
    "../autodl-tmp/models/Qwen2.5-VL-7B-Instruct/",
    use_fast=False,
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(
    "../autodl-tmp/models/Qwen2.5-VL-7B-Instruct/",
    trust_remote_code=True
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "../autodl-tmp/models/Qwen2.5-VL-7B-Instruct/",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
model.enable_input_require_grads()

# ============================

# ============================
train_json_path = "data_vl_grid.json"
with open(train_json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)


train_data = data[:-200]
val_test_data = data[-200:]

with open("data_vl_train_grid.json", "w") as f:
    json.dump(train_data, f)
with open("data_vl_val_grid.json", "w") as f:
    json.dump(val_test_data, f)
with open("data_vl_test_grid.json", "w") as f:
    json.dump(val_test_data, f)


train_ds = Dataset.from_json("data_vl_train_grid.json")
val_ds = Dataset.from_json("data_vl_val_grid.json")
test_ds = Dataset.from_json("data_vl_test_grid.json")

train_dataset = train_ds.map(process_func)
val_dataset = val_ds.map(process_func)

# ============================

# ============================
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    inference_mode=False,
    r=128,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
)

peft_model = get_peft_model(model, lora_config)

# ============================

# ============================
training_args = TrainingArguments(
    output_dir="../autodl-tmp/captcha_ocr_model",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=50,
    save_steps=100,
    eval_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    seed=42,
    data_seed=42,
)


swanlab_callback = SwanLabCallback(
    project="captcha-grid-project",
    experiment_name="qwen2.5-vl-grid",
    config={
        "model": "Qwen2.5-VL-7B-Instruct",
        "dataset_info": "local-grid",
        "train_data_number": len(train_data),
        "lora_rank": 128,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
    },
)


early_stopping = EarlyStoppingCallback(early_stopping_patience=5)


trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback, early_stopping],
)

trainer.train()


best_model_path = trainer.state.best_model_checkpoint
val_peft_model = PeftModel.from_pretrained(model, best_model_path).to("cuda")


with open("data_vl_test_grid.json", "r", encoding='utf-8') as f:
    test_dataset = json.load(f)

test_image_list = []
for item in test_dataset:
    input_image_prompt = item["conversations"][0]["value"]
    origin_image_path = input_image_prompt.split("<tool_response>")[1].split("</tool_call>")[0]

    prompt_text = input_image_prompt.split("<tool_response>")[0].strip()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": origin_image_path},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    response = predict(messages, val_peft_model)
    messages.append({"role": "assistant", "content": f"{response}"})
    print("Prediction:", messages[-1])

    test_image_list.append(swanlab.Image(origin_image_path, caption=response))

swanlab.log({"Prediction": test_image_list})
swanlab.finish()