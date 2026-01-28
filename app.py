import streamlit as st
import logging
import torch
from unsloth import FastLanguageModel

# --- 1. 初期設定 & 警告抑制 ---
# ELYZA(Llama-2)モデル特有の警告を無視し、Unslothの読み込みエラーを回避する設定
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# ページ設定
st.set_page_config(
    page_title="Guardian AI v1",
    page_icon="🛡️",
    layout="wide"
)

# --- 2. モデルのロード (キャッシュ化) ---
# @st.cache_resource を使うことで、ブラウザをリロードしてもモデルを読み込み直さない（爆速化）
@st.cache_resource
def load_model():
    # 学習済みモデルのフォルダパス（リポジトリ直下の 'lora_model' を参照）
    model_name = "lora_model" 
    
    # モデルとトークナイザーの読み込み
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

# --- 3. サイドバー (設定エリア) ---
with st.sidebar:
    st.header("⚙️ システム設定 / Settings")
    
    # モデルロード状態の表示
    with st.status("AIモデル起動中...", expanded=True) as status:
        try:
            model, tokenizer = load_model()
            status.update(label="モデル読み込み完了 (Ready)", state="complete", expanded=False)
        except Exception as e:
            status.update(label="モデル読み込みエラー", state="error")
            st.error(f"モデルが見つかりません: {e}")
    
    st.divider()
    
    # 推論パラメータ調整
    st.subheader("推論パラメータ")
    temperature = st.slider(
        "厳密さ (Temperature)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.1, 
        step=0.1,
        help="0に近いほど論理的で毎回同じ回答をします。上げると創造的になりますが、法務チェックでは0.1〜0.3が推奨です。"
    )
    max_tokens = st.slider(
        "回答の長さ (Max Tokens)", 
        min_value=128, 
        max_value=1024, 
        value=512,
        step=64
    )
    
    st.markdown("---")
    st.caption("Developed by Guardian AI Project")
    st.caption("Base Model: ELYZA-japanese-Llama-2-7b")

# --- 4. プロンプト定義 (Alpaca Format) ---
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# --- 5. メインUI ---
st.title("🛡️ Guardian AI v1")
st.markdown("### IT法務特化型 コンプライアンスチェックAI")
st.info("契約書の条文、仕様書のテキスト、または法的な相談内容を入力してください。AIがリスク判定と修正案を提示します。")

# 画面レイアウト (2カラム)
col1, col2 = st.columns([1, 1])

# 左側：入力エリア
with col1:
    st.subheader("📝 入力 (Input)")
    input_text = st.text_area(
        "解析対象のテキスト",
        height=400,
        placeholder="ここに契約書の条文や相談内容を貼り付けてください。\n\n例：\n損害賠償の請求額は、理由の如何を問わず、本契約に基づき甲が乙に支払った直近1ヶ月分の委託料を上限とする。"
    )
    
    analyze_btn = st.button("リスク判定を実行 (Analyze)", type="primary", use_container_width=True)

# 右側：出力エリア
with col2:
    st.subheader("⚖️ 診断結果 (Result)")
    
    if analyze_btn:
        if not input_text:
            st.warning("⚠️ テキストが入力されていません。")
        else:
            with st.spinner("条項を解析中... (AIが思考しています)"):
                try:
                    # プロンプトの構築
                    prompt = alpaca_prompt.format(
                        "IT法務の専門家として、以下の条項のリスクを判定し、修正案を提示してください。", 
                        input_text, 
                        ""
                    )
                    
                    # トークン化とGPUへの転送
                    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

                    # 推論実行
                    outputs = model.generate(
                        **inputs, 
                        max_new_tokens = max_tokens,
                        use_cache = True,
                        temperature = temperature, 
                    )
                    
                    # 結果のデコードと整形
                    response_text = tokenizer.batch_decode(outputs)[0]
                    # プロンプト部分を除去して回答部分だけ抽出
                    cleaned_response = response_text.split("### Response:\n")[-1].replace(tokenizer.eos_token, "")
                    
                    # 結果表示
                    st.success("解析完了")
                    st.markdown(cleaned_response)
                    
                except Exception as e:
                    st.error(f"エラーが発生しました: {e}")

    else:
        st.info("👈 左側のフォームに入力してボタンを押すと、ここに結果が表示されます。")

# フッター
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
    ※本AIの回答は学習データに基づく予測であり、法的助言（リーガルアドバイス）ではありません。<br>
    最終的な契約判断や紛争解決にあたっては、必ず弁護士等の専門家にご相談ください。
    </div>
    """, 
    unsafe_allow_html=True
)