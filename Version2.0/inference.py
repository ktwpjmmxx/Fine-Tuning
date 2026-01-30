from unsloth import FastLanguageModel
import torch
import os

# --- 設定 ---
# 学習時に保存したDriveのパス
MODEL_PATH = "/content/drive/MyDrive/Llama3_FineTune/lora_model_llama3"

# テストしたい入力
test_instruction = "IT法務コンサルタントとして回答してください。"
test_input = "開発委託契約で、納品物の検収期間を『永続的に』設定したいと言われたが、リスクはあるか？"

# --- 推論実行 ---
print(f"📂 Driveからモデルをロードしています: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    print("❌ モデルが見つかりません。train.pyが正常に完了しているか確認してください。")
    exit()

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_PATH,
    max_seq_length = 4096,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

# プロンプト作成
prompt = f"""<|start_header_id|>system<|end_header_id|>

{test_instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>

{test_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")

print("生成中...")
outputs = model.generate(
    **inputs, 
    max_new_tokens = 512, 
    use_cache = True,
    temperature = 0.1, # 事実重視設定
)

result = tokenizer.batch_decode(outputs)
# 不要な特殊トークンを除去して表示
clean_output = result[0].split("<|start_header_id|>assistant<|end_header_id|>")[-1].replace("<|eot_id|>", "")

print("\n=== 推論結果 ===")
print(clean_output)
print("==================")