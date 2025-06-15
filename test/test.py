import os
import torch
import json
import pandas as pd
from transformers import AutoTokenizer, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, TaskType, PeftModel
from transformers import Qwen2_5_VLForConditionalGeneration


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tokenizer = AutoTokenizer.from_pretrained("model", use_fast=False, trust_remote_code=True)
processor = AutoProcessor.from_pretrained("model", trust_remote_code=True)

val_lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=True,
    r=128,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "model",
    device_map="auto",
    torch_dtype=torch.float16,
    use_cache=False,
    trust_remote_code=True
)

best_model_path = " "

val_peft_model = PeftModel.from_pretrained(model, best_model_path, config=val_lora_config)


with open("data_vl_test.json", "r", encoding='utf-8') as f:
    test_data = json.load(f)

def normalize_predictions(pred):

    return pred.replace("O", "0").replace("o", "0")

def predict(messages, model):

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )


    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)


    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
    ]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)

    return normalize_predictions(output_text[0])

def calculate_accuracy_with_image_comparison(test_data, model):

    correct_sensitive = 0
    correct_insensitive = 0
    total = len(test_data)

    sensitive_predictions = []
    insensitive_predictions = []

    for item in test_data:
        input_image_prompt = item["conversations"][0]["value"]
        actual_text = item["conversations"][1]["value"]


        origin_image_path = input_image_prompt.split("<tool_response>")[1].split("</tool_call>")[0]


        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": origin_image_path},
                    {"type": "text", "text": "Please enter the characters you see on the picture:"},
                ],
            }
        ]


        response = predict(messages, model)


        if response == actual_text:
            correct_sensitive += 1
            sensitive_predictions.append((response, actual_text, origin_image_path))


        if response.lower() == actual_text.lower():
            correct_insensitive += 1
            insensitive_predictions.append((response, actual_text, origin_image_path))


    accuracy_sensitive = correct_sensitive / total
    accuracy_insensitive = correct_insensitive / total
    print(f"sensitive_predictions: {accuracy_sensitive * 100:.2f}%")
    print(f"insensitive_predictions: {accuracy_insensitive * 100:.2f}%")

    sensitive_df = pd.DataFrame(sensitive_predictions, columns=["Predicted", "Actual", "Image Path"])
    insensitive_df = pd.DataFrame(insensitive_predictions, columns=["Predicted", "Actual", "Image Path"])


    sensitive_df.to_csv("sensitive_predictions.csv", index=False)
    insensitive_df.to_csv("insensitive_predictions.csv", index=False)


calculate_accuracy_with_image_comparison(test_data, val_peft_model)
