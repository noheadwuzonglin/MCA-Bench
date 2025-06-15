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
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    inference_mode=True,
    r=128,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
)

val_peft_model = PeftModel.from_pretrained(model, best_model_path, config=val_lora_config)


with open("data_gridexchange_test.json", "r", encoding='utf-8') as f:
    test_data = json.load(f)


def is_point_in_bbox(x, y, bbox):
    """
    Check if a point (x, y) is inside the bounding box defined by bbox.
    bbox format: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def validate_predictions(predicted_points, true_labels):
    """
    Validate whether the predicted points match the true labels.
    Ignores the order of the predicted points.

    predicted_points: List of predicted points (as string or tuple, e.g., '7 3')
    true_labels: List of true points (as string or tuple, e.g., '7 3')
    """
    # Sort both predicted points and true labels to avoid order mismatch
    predicted_points_sorted = sorted(predicted_points)
    true_labels_sorted = sorted(true_labels)

    # Check if both sorted lists match
    return predicted_points_sorted == true_labels_sorted


def predict(messages, model):

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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


def calculate_accuracy_with_image_comparison(test_data, model):

    correct = 0
    total = len(test_data)

    predictions = []

    for item in test_data:
        input_image_prompt = item["conversations"][0]["value"]
        actual_text = item["conversations"][1]["value"]


        user_val = item["conversations"][0]["value"]
        file_path = user_val.split("<tool_response>")[1].split("</tool_response>")[0]


        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": file_path, "resized_height": 224, "resized_width": 224},
                {"type": "text", "text": actual_text},
            ]
        }]


        response = predict(messages, model)


        predicted_points = response.split()


        true_labels = actual_text.split()


        validation_result = validate_predictions(predicted_points, true_labels)


        print(f"Image Path: {file_path}")
        print(f"Predicted Points: {predicted_points}")
        print(f"True Labels: {true_labels}")
        print(f"Validation Passed: {validation_result}\n")


        predictions.append({
            "Image Path": file_path,
            "Predicted Points": predicted_points,
            "True Labels": true_labels,
            "Validation Passed": validation_result
        })


        if validation_result:
            correct += 1

    accuracy = correct / total
    print(f"pass rate: {accuracy * 100:.2f}%")

    predictions_df = pd.DataFrame(predictions)

    predictions_df.to_csv("predictions.csv", index=False)


# 调用函数并生成预测结果表格
calculate_accuracy_with_image_comparison(test_data, val_peft_model)
