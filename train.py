from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# 1. Configuration
max_seq_length = 4096 # 法務文書は長いので、余裕があれば2048から4096へ拡張（エラーが出るなら2048に戻してください）
dtype = None 
load_in_4bit = True 

# 2. Load Model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "elyza/ELYZA-japanese-Llama-2-7b-instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 3. LoRA Adapters Config
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    lora_dropout = 0.05, # 少しドロップアウトを入れて過学習を防ぐ
    bias = "none",
    use_gradient_checkpointing = True,
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# 4. Data Formatting (EOSトークン対応)
# AIに「ここで話終わり」を教えるための重要な処理です
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # 終了トークン

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # 必ず末尾に EOS_TOKEN を付ける
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

# 5. Load Dataset (ファイル名を最新版に変更)
dataset = load_dataset("json", data_files="traindata.jsonl", split="train")
dataset = dataset.map(formatting_prompts_func, batched = True)

# 6. Training Arguments
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Short sequences handling
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # max_steps = 60, # 旧設定：ステップ数固定は廃止
        num_train_epochs = 3, # 新設定：データ全体を3周しっかり学習させる
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# 7. Train
trainer_stats = trainer.train()

# 8. Save Model (LoRA Adapter)
model.save_pretrained("lora_model") # 名前は任意
tokenizer.save_pretrained("lora_model")
print("✅ Training Completed and Model Saved!")