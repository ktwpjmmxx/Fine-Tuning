# Guardian AI v1 (Fine-Tuned)

IT法務コンサルタント特化型LLM「Guardian AI」の第一世代（v1）モデルのリポジトリです。
[cite_start]ELYZA-japanese-Llama-2-7b-instruct をベースに、計2,819件の高品質なデータセットを用いてファインチューニングを行い、IT実務における法的リスク判定と修正提案能力を強化しました [cite: 1, 3]。

## プロジェクト概要

[cite_start]本プロジェクトは、開発者やプロダクトマネージャーが直面する法的リスク（契約条項、UI/UX仕様、広告表現）を、IT法務の観点から判定し、具体的な修正案を提示するAIモデルの構築を目的としています [cite: 1, 3]。

[cite_start]v1アップデートでは、初版の課題であった「回答フォーマットの揺れ」を完全に解消し、法務実務で頻出する「裁判管轄」「下請法」「労働法（偽装請負）」等の専門領域を大幅に強化しました [cite: 3]。

## 学習メトリクス (v1)

* [cite_start]**Dataset Size**: 2,819 samples (Unified & Polished) [cite: 3]
* [cite_start]**Base Model**: elyza/ELYZA-japanese-Llama-2-7b-instruct [cite: 3]
* [cite_start]**Training Loss**: 0.6030 (3.0 Epochs) [cite: 2]
* [cite_start]**Training Method**: LoRA (4-bit Quantization / Unsloth optimized) [cite: 3]
* [cite_start]**Training Hardware**: NVIDIA Tesla L4 / T4 (Google Colab Environment) [cite: 3]

## 実装機能（学習済みスキル）

### 1. 契約書リスク判定・修正提案
[cite_start]IT開発契約における不利な条項を特定し、受託者側に有利、または公平な代替条文を提示します [cite: 1, 3]。
* [cite_start]著作権の帰属および著作者人格権の不行使条項の是正 [cite: 3]
* [cite_start]裁判管轄（専属的合意管轄）のリスク指摘と東京地裁への集約 [cite: 3]
* [cite_start]損害賠償制限の範囲および免責事項の適正化 [cite: 3]

### 2. 労働・下請法コンプライアンス
[cite_start]IT業界特有の商習慣に潜む法的リスクを検知します [cite: 3]。
* [cite_start]偽装請負（指揮命令権の独立性）の判定と修正 [cite: 1, 3]
* [cite_start]下請法（支払遅延、不当な減額、買いたたき）の違反検知 [cite: 1, 3]
* [cite_start]フリーランス保護新法に基づくハラスメント対策および中途解除予告の徹底 [cite: 3]

### 3. プロダクト・マーケティング監査
[cite_start]ユーザーインターフェースや広告表現におけるリーガルチェックを行います [cite: 3]。
* [cite_start]景品表示法（優良誤認、不実証広告）のリスク判定 [cite: 1, 3]
* [cite_start]UIにおけるダークパターンの検知 [cite: 3]
* [cite_start]薬機法、特定商取引法に基づく表現の修正提案 [cite: 3]

### 4. 安全装置（アウト・オブ・ドメイン拒絶）
[cite_start]IT法務以外の専門外の相談（離婚、相続、刑事事件、税務等）に対し、プロフェッショナルな品格を保ちつつ回答を拒絶し、適切な専門家（税理士、専門弁護士、警察等）へ誘導する機能を備えています [cite: 1, 3]。

## 推論サンプル

[cite_start]モデルは以下の4項目による構造化出力を安定して行います [cite: 1, 3]。

---
**Q: 準委任契約（SES）において、発注者が弊社のエンジニアに直接チャットで作業指示を出す旨の条項を追加したいと言っています。問題ありますか？**

[cite_start]**リスクレベル**: High [cite: 1]
[cite_start]**該当法**: 労働者派遣法、職業安定法第44条（労働者供給事業の禁止） [cite: 1]
[cite_start]**理由**: 準委任契約や請負契約において、発注者が受託者の従業員に直接の指揮命令を行うことは「偽装請負」とみなされます。これは法的制約だけでなく、行政処分の対象となる重大なコンプライアンス違反です [cite: 1]。
[cite_start]**修正案**: 「第○条 乙は、本件業務の遂行に関して、自らの責任において乙の従業員を指揮命令する。甲からの業務上の要望等は、乙が選任した責任者を通じて行うものとし、乙の従業員個人へ直接指示を行わないものとする。」 [cite: 1]
---

## 技術スタック

* [cite_start]言語モデル: Llama-2-7b-based (ELYZA Instruct) [cite: 3]
* [cite_start]最適化ライブラリ: Unsloth [cite: 3]
* [cite_start]学習手法: QLoRA (Rank=16, Alpha=32) [cite: 3]
* [cite_start]評価指標: Training Loss および 推論テストによるフォーマット遵守率 [cite: 3]

## ディレクトリ構成

.
├── dataset/
│   └── traindata.jsonl  # 最終研磨済みデータセット (2,819件)
├── logs/
│   ├── train_log.txt                 # v1 学習ログ (Final Loss: 0.6030)
│   └── inference_log.txt             # v1 推論テスト結果
├── train.py                          # Unslothを用いた学習用スクリプト
├── inference.py                      # 推論検証用スクリプト
└── README.md

## 今後の展望 (Future Roadmap)

本モデルはv1として基本的な法的判断能力を備えていますが、今後は以下のアップデートを予定しています。
* 特定の業界約款（SaaS、官公庁システム開発等）に特化した専門特化型アダプタの開発。
* 契約書比較（Diff）機能と連動した、自動修正コメント生成機能の実装。

## 免責事項 (Disclaimer)

本モデルが出力する回答は学習データに基づく予測であり、法的助言を構成するものではありません。実際の契約実務や紛争解決にあたっては、必ず資格を有する弁護士等の専門家に相談してください。本モデルの利用により生じた損害について、開発者は一切の責任を負いません。