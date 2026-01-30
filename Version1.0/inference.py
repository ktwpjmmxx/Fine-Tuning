import logging
# 【重要】ELYZA(Llama-2)の古い重み形式による無害な警告を無視し、Unslothの停止を回避する設定
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

from unsloth import FastLanguageModel
import torch

# 1. Configuration
max_seq_length = 2048 # 学習時と同じ設定
dtype = None
load_in_4bit = True

# 2. Load Model (学習済みのモデルをロード)
# "lora_model" フォルダが学習によって生成されているはずです
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "lora_model", 
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model)

# 3. Prompt Template (学習時と同じAlpacaフォーマット)
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# 4. Inference Function
def guardian_ai_check(instruction, input_text):
    inputs = tokenizer(
        [
            alpaca_prompt.format(
                instruction, # Instruction
                input_text,  # Input
                "",          # Response (空にして続きを書かせる)
            )
        ], return_tensors = "pt").to("cuda")

    outputs = model.generate(
        **inputs, 
        max_new_tokens = 512, # 回答の長さ上限
        use_cache = True,
        temperature = 0.1,    # 法務なので厳密さを重視して低めに設定
    )
    
    # プロンプト部分を除去して回答だけを取り出す
    response = tokenizer.batch_decode(outputs)
    cleaned_response = response[0].split("### Response:\n")[-1].replace(tokenizer.eos_token, "")
    return cleaned_response

# --- テスト実行 ---
print("⚖️ 推論テスト開始...\n")

# テストケース1: 典型的なリスク条項
input_text = "損害賠償の請求額は、理由の如何を問わず、本契約に基づき甲が乙に支払った直近1ヶ月分の委託料を上限とする。"
print(f"Q: {input_text}")
print("-" * 30)
print(guardian_ai_check("IT法務の専門家として、以下の条項のリスクを判定し、修正案を提示してください。", input_text))
print("=" * 30)

# テストケース2: 専門外（拒絶テスト）
input_text = "離婚調停の進め方について教えてください。"
print(f"Q: {input_text}")
print("-" * 30)
print(guardian_ai_check("IT法務の専門家として、以下の相談に回答してください。専門外の場合は適切に断ってください。", input_text))