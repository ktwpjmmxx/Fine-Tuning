import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import os

# --- 設定 ---
# 保存先ディレクトリ（Drive直下を指定）
OUTPUT_DIR = "/content/drive/MyDrive/Llama3_FineTune/lora_model_llama3"
DATA_FILE = "traindata_v2.jsonl"
MODEL_NAME = "elyza/Llama-3-ELYZA-JP-8b"

max_seq_length = 4096 # Llama-3は長文対応なので4096推奨
dtype = None 
load_in_4bit = True 

print("モデルをロードしています...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# LoRAアダプター設定
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none",
    use_gradient_checkpointing = True,
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# --- プロンプトフォーマット関数 (Llama-3仕様) ---
llama3_prompt = """<|start_header_id|>system<|end_header_id|>

{}<|eot_id|><|start_header_id|>user<|end_header_id|>

{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{}<|eot_id|>"""

EOS_TOKEN = tokenizer.eos_token 

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Instructionが空の場合のデフォルト値を設定
        if not instruction:
            instruction = "あなたはIT法務の専門家として、ユーザーの質問に法的に正確かつ適切に回答してください。"
            
        text = llama3_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

# データセットのロード
print(f"📂 データセットを読み込んでいます: {DATA_FILE}")
dataset = load_dataset("json", data_files=DATA_FILE, split="train")
dataset = dataset.map(formatting_prompts_func, batched = True)

# --- 学習実行 ---
print("学習を開始します...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, 
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 2, # データ量が多いので2epochで十分
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = OUTPUT_DIR, # Driveのパスに出力
        save_strategy = "steps", # 途中経過も保存したい場合
        save_steps = 100,        # 100ステップごとに保存
    ),
)

trainer_stats = trainer.train()

# --- 最終保存 ---
print(f"💾 モデルをDriveに保存しています: {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✅ すべての処理が完了しました。")