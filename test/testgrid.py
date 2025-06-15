import os
import json
import torch
from peft import LoraConfig, TaskType, PeftModel
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from modelscope import snapshot_download
from PIL import Image
from typing import List

###################################

###################################

from transformers import Qwen2_5_VLForConditionalGeneration

#########################

#########################
model_path = "../autodl-tmp/models/Qwen2.5-VL-7B-Instruct/"
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    use_fast=False,
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(
    model_path,
    trust_remote_code=True
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
model.enable_input_require_grads()


best_model_checkpoint = " "

val_lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    inference_mode=True,
    r=128,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
)


val_peft_model = PeftModel.from_pretrained(
    model,
    best_model_checkpoint,
    config=val_lora_config
).to("cuda")


########################

########################
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


########################

########################
with open("data_vl_test_grid.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

num_samples = len(test_data)
num_correct = 0

for idx, item in enumerate(test_data):

    conversation = item["conversations"]
    user_content = conversation[0]["value"]
    gold_answer = conversation[1]["value"].strip()


    image_path = user_content.split("<tool_response>")[1].split("</tool_call>")[0].strip()


    prompt_text = user_content.split("<tool_response>")[0].strip()


    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
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


    prediction = predict(messages, val_peft_model)

    gold_set = set([x.strip() for x in gold_answer.split(",")])
    pred_set = set([x.strip() for x in prediction.split(",")])

    if gold_set == pred_set:
        num_correct += 1


    print(f"[{idx + 1}/{num_samples}] Gold: {gold_answer}, Pred: {prediction}, ",
          "Correct" if gold_set == pred_set else "Wrong")


accuracy = num_correct / num_samples
print("\n======================")
print(f"{num_samples}")
print(f"{num_correct}")
print(f"passrate: {accuracy:.2%}")
