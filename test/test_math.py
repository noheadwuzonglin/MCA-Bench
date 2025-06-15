import torch
from transformers import AutoTokenizer, AutoProcessor
from peft import PeftModel, LoraConfig, TaskType
from datasets import Dataset
import json
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

    if "attention_mask" not in model_inputs:
        model_inputs["attention_mask"] = torch.ones_like(model_inputs["input_ids"])

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                     zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


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


        correct_answer = item["conversations"][1]["value"].strip()


        print(f"Image: {origin_image_path}")
        print(f"Prediction: {response.strip()}")
        print(f"True Label: {correct_answer}")


        if response.strip() == correct_answer:
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


with open("data_math_test.json", "r", encoding='utf-8') as f:
    test_dataset = json.load(f)


accuracy = evaluate_model(test_dataset, val_peft_model, tokenizer)
