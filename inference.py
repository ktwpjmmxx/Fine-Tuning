import json
import torch
from unsloth import FastLanguageModel

# 1. 設定の読み込み
with open("config.json", "r") as f:
    config = json.load(f)

# 学習済みのモデルパス（outputsフォルダの中のモデル名を指定）
model_path = os.path.join(config["output_dir"], config["new_model_name"])

print(f"📂 モデルをロード中: {model_path}")

# 2. モデルロード
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path, # ここで学習済みアダプターを指定
    max_seq_length=config["max_seq_length"],
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

# 3. プロンプトテンプレート
alpaca_prompt = """<s>[INST] {instruction}
{input} [/INST]
"""

# 4. 推論実行関数
def generate_response(instruction, input_text):
    prompt = alpaca_prompt.format(instruction=instruction, input=input_text)
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        use_cache=True,
        temperature=0.3, # 法務なので少し低め（ランダム性を抑える）
        top_p=0.9,
    )
    result = tokenizer.batch_decode(outputs)
    # 生成部分のみ抽出する処理（簡易版）
    return result[0].split("[/INST]")[1].replace("</s>", "").strip()

# --- テスト実行 ---
if __name__ == "__main__":
    test_instruction = "IT法務コンサルタントとして、プロダクト仕様に関連する法的リスクを判定し、実務的な修正案を提示してください。"
    test_input = "退会ボタンをあえて見つけにくい場所に配置して、ユーザーの離脱を防ぎたいです。"
    
    print("\n--- Input ---")
    print(test_input)
    print("\n--- Output ---")
    print(generate_response(test_instruction, test_input))