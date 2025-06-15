import torch
from transformers import AutoTokenizer, AutoProcessor
from peft import PeftModel, LoraConfig, TaskType
import json
import re
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    EarlyStoppingCallback
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    inference_mode=False,
    r=128,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
)


def predict(messages, model, tokenizer):
    device = "cuda"


    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(device)


    if torch.any(torch.isnan(model_inputs['input_ids'])) or torch.any(torch.isinf(model_inputs['input_ids'])):
        print(f"NaN or Inf detected in input_ids: {model_inputs['input_ids']}")
        return "Error: Invalid input"


    if "attention_mask" not in model_inputs:
        model_inputs["attention_mask"] = torch.ones_like(model_inputs["input_ids"])


    logits = model.model(model_inputs['input_ids']).logits
    if torch.any(torch.isnan(logits)) or torch.any(torch.isinf(logits)):
        print(f"NaN or Inf detected in logits: {logits}")
        return "Error: Invalid logits"

    try:
        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512, do_sample=True, top_p=0.95,
                                       temperature=0.7)
    except RuntimeError as e:
        print(f"Error during generation: {e}")
        return "Error during generation."


    print("generated_ids:", generated_ids)

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                     zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


def extract_coordinates_from_response(response):

    coords = []
    pattern = r"(\d+)\s?\((\d+),(\d+)\)"
    matches = re.findall(pattern, response)
    for match in matches:
        coords.append((match[0], int(match[1]), int(match[2])))
    return coords


def extract_true_regions(item):

    true_regions = []
    regions = item.get("regions", {})
    if not regions:
        print(f"Warning: No regions found in item {item['id']}")
    for region_id, coordinates in regions.items():

        print(f"Region ID: {region_id}, Coordinates: {coordinates}")


        if len(coordinates) == 4:
            true_regions.append(
                (region_id, coordinates[0], coordinates[1], coordinates[2], coordinates[3]))  # (ID, x1, y1, x2, y2)
        else:

            print(f"Warning: Invalid region data for {region_id}. Expected 4 coordinates, got {len(coordinates)}.")

    return true_regions


def is_prediction_correct(predicted_coords, true_regions):

    if len(predicted_coords) != len(true_regions):
        print(f"Mismatch in number: predicted {len(predicted_coords)} points, but {len(true_regions)} regions.")
        return False

    used = [False] * len(true_regions)
    for _, px, py in predicted_coords:
        matched = False
        for i, (_, x1, y1, x2, y2) in enumerate(true_regions):
            if not used[i] and x1 <= px <= x2 and y1 <= py <= y2:
                used[i] = True
                matched = True
                break
        if not matched:

            return False

    return True


def evaluate_model(test_dataset, model, tokenizer):
    correct_predictions = 0
    total_predictions = len(test_dataset)

    for item in test_dataset:
        input_image_prompt = item["conversations"][0]["value"]
        origin_image_path = input_image_prompt.split("<tool_response>")[1].split("</tool_response>")[0]


        prompt_text = input_image_prompt.split("<tool_response>")[0].strip()


        messages = [
            {
                "role": "user",
                "content": f"{prompt_text} <tool_response>{origin_image_path}</tool_response>",
            }
        ]

        response = predict(messages, model, tokenizer)
        messages.append({"role": "assistant", "content": f"{response}"})


        predicted_coords = extract_coordinates_from_response(response)


        true_regions = extract_true_regions(item)

        print(f"Image: {origin_image_path}")
        print(f"Prediction: {predicted_coords}")
        print(f"True Regions: {true_regions}")


        if is_prediction_correct(predicted_coords, true_regions):
            correct_predictions += 1


    accuracy = correct_predictions / total_predictions
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    return accuracy



tokenizer = AutoTokenizer.from_pretrained(
    "../autodl-tmp/models/Qwen2.5-VL-7B-Instruct/",
    use_fast=False
)
processor = AutoProcessor.from_pretrained(
    "../autodl-tmp/models/Qwen2.5-VL-7B-Instruct/"
)


model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "../autodl-tmp/models/Qwen2.5-VL-7B-Instruct/",
    device_map="auto",
    torch_dtype=torch.float16
)


best_model_path = " "
val_peft_model = PeftModel.from_pretrained(model, best_model_path, config=lora_config).to("cuda")

test_json_path = "data_click_math_test.json"
with open(test_json_path, "r", encoding='utf-8') as f:
    test_dataset = json.load(f)


accuracy = evaluate_model(test_dataset, val_peft_model, tokenizer)
