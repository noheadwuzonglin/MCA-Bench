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


best_model_path = " "

val_lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=True,
    r=128,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
)

val_peft_model = PeftModel.from_pretrained(model, best_model_path, config=val_lora_config)


with open("ud_test.json", "r", encoding='utf-8') as f:
    test_data = json.load(f)


def is_point_in_bbox(x, y, bbox):
    """
    Check if a point (x, y) is inside the bounding box defined by bbox.
    bbox format: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def validate_predictions(predicted_points, regions):
    """
    Validate whether the predicted points fall within any of the regions (bounding boxes).
    Also ensures that the number of predicted points matches the number of regions.

    predicted_points: List of (x, y) tuples
    regions: List of regions, each defined as [x1, y1, x2, y2]
    """
    # Check if the number of predicted points matches the number of regions
    if len(predicted_points) != len(regions):
        print(
            f"Prediction count does not match regions count. Predicted: {len(predicted_points)}, Regions: {len(regions)}")
        return False  # Validation failed if counts don't match

    for point in predicted_points:
        found = False
        for region in regions:
            if is_point_in_bbox(point[0], point[1], region):
                found = True
                break
        if not found:
            return False  # If any point is not within any region, return False
    return True  # All points are inside the bounding boxes


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

    predicted_points = []
    for coords in output_text[0].split(';'):
        coords = coords.strip()
        if coords:

            x, y = coords.strip('()').split(',')
            predicted_points.append((int(x), int(y)))

    return predicted_points


def calculate_accuracy_with_image_comparison(test_data, model):

    correct = 0
    total = len(test_data)

    predictions = []

    for item in test_data:
        input_image_prompt = item["conversations"][0]["value"]
        actual_text = item["conversations"][1]["value"]


        user_val = item["conversations"][0]["value"]
        file_path = user_val.split("<tool_response>")[1].split("</tool_response>")[0]

        # 构造多模态消息
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": file_path, "resized_height": 224, "resized_width": 224},
                {"type": "text", "text": "Pick out all the upside down letters in the picture."},
            ]
        }]


        predicted_points = predict(messages, model)


        regions = item["regions"]

        validation_result = validate_predictions(predicted_points, regions)


        print(f"Image Path: {file_path}")
        print(f"Predicted Points: {predicted_points}")
        print(f"Regions (Ground Truth): {regions}")
        print(f"Validation Passed: {validation_result}\n")


        predictions.append({
            "Image Path": file_path,
            "Predicted Points": predicted_points,
            "Validation Passed": validation_result
        })


        if validation_result:
            correct += 1


    accuracy = correct / total
    print(f"pass rate: {accuracy * 100:.2f}%")


    predictions_df = pd.DataFrame(predictions)


    predictions_df.to_csv("predictions.csv", index=False)



calculate_accuracy_with_image_comparison(test_data, val_peft_model)
