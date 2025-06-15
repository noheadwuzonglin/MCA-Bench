import os, json, random, torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments, Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType


BASE_JSON  = "math_qa.json"
TRAIN_JSON = "train.json"
TEST_JSON  = "test.json"
MODEL_PATH = "../autodl-tmp/models/Qwen2.5-VL-7B-Instruct/"
OUTPUT_DIR = "../autodl-tmp/qa"
SEED       = 42
MAX_LEN    = 1024
R_LORA     = 64
ALPHA      = 16
DROPOUT    = 0.05
BATCH_SIZE = 8
EPOCHS     = 10
SAVE_STEPS = 100
EVAL_STEPS = 100
LOG_STEPS  = 50

print("📦 Loading tokenizer & processor & VL model …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, device_map="auto", torch_dtype=torch.float16
)
base_model.enable_input_require_grads()
print("✅ Model loaded.\n")


def split_and_save():
    with open(BASE_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    random.seed(SEED)
    random.shuffle(data)
    n = len(data)
    n_tr = int(0.8 * n)
    train = data[:n_tr]
    test  = data[n_tr:]

    with open(TRAIN_JSON, 'w', encoding='utf-8') as fw:
        json.dump(train, fw, ensure_ascii=False, indent=2)
    with open(TEST_JSON, 'w', encoding='utf-8') as fw:
        json.dump(test,  fw, ensure_ascii=False, indent=2)
    print(f"Split data → {len(train)} train, {len(test)} test/val.\n")


def process_func(example):
    prompt = example["conversations"][0]["value"].strip()
    answer = example["conversations"][1]["value"].strip()

    messages = [{"role":"user","content":[{"type":"text","text":prompt}]}]
    text     = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        padding=True,
        return_tensors="pt"
    )

    input_ids      = inputs["input_ids"][0].tolist()
    attention_mask = inputs["attention_mask"][0].tolist()

    resp = tokenizer(answer, add_special_tokens=False)
    input_ids      += resp["input_ids"] + [tokenizer.pad_token_id]
    attention_mask += resp["attention_mask"] + [1]

    prompt_enc = tokenizer(text, add_special_tokens=False)
    prompt_len = len(prompt_enc["input_ids"])
    labels = [-100]*prompt_len + resp["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > MAX_LEN:
        input_ids      = input_ids[:MAX_LEN]
        attention_mask = attention_mask[:MAX_LEN]
        labels         = labels[:MAX_LEN]
    return {
        "input_ids":      torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels":         torch.tensor(labels, dtype=torch.long),
    }

if __name__ == "__main__":

    split_and_save()


    print("🔄 Building Dataset & mapping …")
    train_ds = Dataset.from_json(TRAIN_JSON).map(
        process_func, remove_columns=None, desc="Map train", num_proc=4
    )

    test_ds  = Dataset.from_json(TEST_JSON).map(
        process_func, remove_columns=None, desc="Map test", num_proc=4
    )
    print("✅ Data ready.\n")


    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        inference_mode=False,
        r=R_LORA, lora_alpha=ALPHA, lora_dropout=DROPOUT, bias="none"
    )
    model = get_peft_model(base_model, lora_cfg)


    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        logging_steps=LOG_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        fp16=True,
        seed=SEED,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    trainer       = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator
    )


    print("🚀 Starting training …")
    trainer.train()

    model.save_pretrained(os.path.join(OUTPUT_DIR, "final_peft"))
    print(f"🎉 Done! Weights at {OUTPUT_DIR}/final_peft")
