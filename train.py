import json
import os
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# 1. 設定の読み込み
with open("config.json", "r") as f:
    config = json.load(f)

# 2. モデルとトークナイザーのロード (Unsloth)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=config["model_name"],
    max_seq_length=config["max_seq_length"],
    dtype=None,
    load_in_4bit=config["load_in_4bit"],
)

# 3. LoRAアダプターの設定
model = FastLanguageModel.get_peft_model(
    model,
    r=config["lora_r"],
    target_modules=config["target_modules"],
    lora_alpha=config["lora_alpha"],
    lora_dropout=config["lora_dropout"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# 4. データセットの準備 (Promptのフォーマット)
# Elyzaのプロンプト形式: <s>[INST] {instruction} {input} [/INST] {output} </s>
alpaca_prompt = """<s>[INST] {instruction}
{input} [/INST]
{output} </s>"""

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction=instruction, input=input, output=output)
        texts.append(text)
    return {"text": texts}

dataset = load_dataset("json", data_files=config["dataset_path"], split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

# 5. 再開(Resume)の判定ロジック
# outputsディレクトリ内にcheckpointフォルダがあるか探す
checkpoints = [d for d in os.listdir(config["output_dir"]) if d.startswith("checkpoint-")]
if checkpoints:
    # 数字部分を取り出してソートし、最新のものを特定
    checkpoints.sort(key=lambda x: int(x.split('-')[1]))
    latest_checkpoint = os.path.join(config["output_dir"], checkpoints[-1])
    print(f"🔄 前回の続きから再開します: {latest_checkpoint}")
    resume_from_checkpoint = latest_checkpoint
else:
    print("新規トレーニングを開始します")
    resume_from_checkpoint = False

# 6. トレーナーの設定
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=config["max_seq_length"],
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        warmup_steps=5,
        num_train_epochs=config["num_train_epochs"],
        learning_rate=config["learning_rate"],
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=config["logging_steps"],
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=config["output_dir"],
        save_strategy="steps",      # ステップごとに保存
        save_steps=config["save_steps"],       # configで指定した頻度
        save_total_limit=2,         # Drive容量圧迫を防ぐため、最新2つだけ残す
    ),
)

# 7. 学習実行
trainer_stats = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

# 8. 最終モデルの保存
print("💾 学習完了。モデルを保存します...")
model.save_pretrained(os.path.join(config["output_dir"], config["new_model_name"]))
tokenizer.save_pretrained(os.path.join(config["output_dir"], config["new_model_name"]))
print("✅ すべて完了しました！")