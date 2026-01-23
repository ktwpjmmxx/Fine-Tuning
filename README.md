# Guardian AI (Fine-Tuning)

IT法務コンサルタント特化型LLM「Guardian AI」のファインチューニング用リポジトリです。
ELYZA-japanese-Llama-2-7b-instruct をベースに、IT法務、契約書チェック、広告リスク判定、仕様相談などに特化した学習を行っています。

## プロジェクト概要

このプロジェクトは、開発者やPMが直面する法的リスク（契約、UI/UX仕様、広告表現）を、IT法務の観点から判定・修正提案するAIモデルを作成することを目的としています。

### 主な機能（学習済みスキル）
* 契約書チェック: 不利な条項（著作権、損害賠償、SES契約等）のリスク判定と修正案提示。
* プロダクト仕様相談: UI/UXにおけるダークパターン、景表法、個人情報保護法のリスク判定。
* マーケティングチェック: 広告表現におけるステマ規制、薬機法、優良誤認等のリスク判定。
* 拒否スキル: 専門外（離婚、刑事事件など）の相談に対する適切な拒否と誘導。

## 技術スタック

* Base Model: elyza/ELYZA-japanese-Llama-2-7b-instruct
* Library: Unsloth (Fast & Memory Efficient)
* Method: LoRA (4-bit Quantization)
* Environment: Google Colab (Free Tier) with Drive Resume capability

## ディレクトリ構成

.
├── dataset/
│   └── traindata_cleaned.jsonl  # 学習データセット（約2,300件）
├── config.json                  # 学習パラメータ設定
├── train.py                     # 学習実行スクリプト（中断箇所からの自動再開機能付き）
├── inference.py                 # 推論テスト用スクリプト
└── README.md

## データセットについて

traindata_cleaned.jsonl には、Geminiを用いて生成・キュレーションされた高品質な 約2,300件 のインストラクションデータが含まれています。

* Contract (契約書): ~300件
* UI/UX (仕様): ~720件
* Marketing (広告): ~520件
* Refusal (守備範囲外): ~290件
* Basic/General: ~480件

## 使い方 (Google Colab)

1. Google Driveに本リポジトリのフォルダを配置します。
2. Colabで以下のコマンドを実行し、環境を構築します。

```python
# 依存ライブラリのインストール
!pip install "unsloth[colab-new] @ git+[https://github.com/unslothai/unsloth.git](https://github.com/unslothai/unsloth.git)"
!pip install --no-deps "xformers<0.0.26" "trl<0.9.0" peft accelerate bitsandbytes
```

3. 学習を開始します（outputs/ フォルダにチェックポイントが保存されます）。

```bash
python train.py
```

4. 推論をテストします。

```bash
python inference.py
```

## 免責事項

本モデルが出力する法的アドバイスは、AIによる推論結果であり、弁護士による法的助言を代替するものではありません。実務での利用においては、必ず専門家の確認を経て利用してください。

## License

Apache License 2.0