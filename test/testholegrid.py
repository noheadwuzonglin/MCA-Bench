import os
import torch
import json
import pandas as pd
from transformers import AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, TaskType, PeftModel
from qwen_vl_utils import process_vision_info


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ============================

# ============================
tokenizer = AutoTokenizer.from_pretrained(
    "../autodl-tmp/models/Qwen2.5-VL-7B-Instruct/", use_fast=False
)
processor = AutoProcessor.from_pretrained(
    "../autodl-tmp/models/Qwen2.5-VL-7B-Instruct/"
)
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "../autodl-tmp/models/Qwen2.5-VL-7B-Instruct/",
    device_map="auto",
    torch_dtype=torch.float16
)

# ============================

# ============================
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    inference_mode=True,   # 推理时打开
    r=64,                  # 与训练时一致
    lora_alpha=8,          # 与训练时一致
    lora_dropout=0.05,
    bias="none",
)

best_checkpoint = "../autodl-tmp/captcha_holegrid_model/checkpoint-7500"  # 替换为你的最佳 checkpoint 路径
model = PeftModel.from_pretrained(
    base_model, best_checkpoint, config=lora_config
).to("cuda")

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
    # trim prompt
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
def validate_predictions(predicted_points, true_labels):
    pred = sorted(predicted_points)
    truth = sorted(true_labels)
    return pred == truth

# ============================

# ============================
def calculate_accuracy(test_data, model):
    total = len(test_data)
    correct = 0
    records = []

    for item in test_data:

        user_val = item["conversations"][0]["value"]

        file_path = user_val.split("<tool_response>")[1].split("</tool_call>")[0]

        prompt_text = user_val.split("<tool_response>")[0].strip()


        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": file_path, "resized_height": 224, "resized_width": 224},
                {"type": "text",  "text": prompt_text},
            ]
        }]

        response = predict(messages, model)

        predicted = response.replace(",", " ").split()
        true_labels = item["conversations"][1]["value"].replace(",", " ").split()

        passed = validate_predictions(predicted, true_labels)
        if passed:
            correct += 1

        records.append({
            "Image Path": file_path,
            "Prompt": prompt_text,
            "Predicted": predicted,
            "True": true_labels,
            "Passed": passed
        })

        print(f"{file_path}  →  pred={predicted}, true={true_labels}, pass={passed}")

    accuracy = correct / total * 100
    print(f"\nOverall Accuracy: {accuracy:.2f}%")


    df = pd.DataFrame(records)
    df.to_csv("predictions.csv", index=False)
    print("Saved detailed results to predictions.csv")

# ============================

# ============================
if __name__ == "__main__":
    with open("data_vl_test_holegrid.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
    calculate_accuracy(test_data, model)
