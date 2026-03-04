#### LLM 基础架构与训练范式总结
##### 1. Encoder-only 架构
- 代表模型：BERT, RoBERTa, BGE (Embedding 模型)
- 核心特点：双向注意力 (Bidirectional Attention)。每个 token 都能看到上下文所有 token，适合需要全局理解的任务（如分类、实体识别、语义匹配）。
- BERT 训练 (MLM)：
  - Masked Language Modeling：随机 Mask 15% 的 token（如 `A [MASK] C D`）。
  - 目标：利用上下文预测被 Mask 的词（预测 `B`）。
  - 输出：在 Encoder 的输出层，直接对 Mask 位置的向量做分类预测。
  - 优势：双向视野带来了极强的上下文理解能力，是判别式任务（Discriminative Tasks）的王者。
- BGE 训练 (RetroMAE)：
  - 动机：BERT 的 `[CLS]` 向量原生质量不高，需要特化训练以适用于检索。
  - 预训练 (RetroMAE)：引入一个轻量级 Decoder（仅用于预训练，推理时丢弃）。
    - Encoder：输入被中度 Mask 的句子，输出 `[CLS]` 向量 h。
    - Decoder：输入 h + 被重度 Mask 的句子，尝试还原原句。
    - 关键点：Decoder 只能通过 h 获取语义信息，强迫 Encoder 将全句语义极致压缩进 h。
  - 微调 (Contrastive Learning)：使用有监督数据（Query-Passage 对），拉近正样本距离，推远负样本（特别是 Hard Negatives），进一步优化检索性能。
##### 2. Encoder-Decoder 架构
- 代表模型：T5, BART, UL2, BLIP (早期的多模态)
- 核心特点：Cross-Attention (交叉注意力)。Encoder 负责理解（双向），Decoder 负责生成（单向）。Decoder 的每一层都会通过 Cross-Attention “回头看” Encoder 的完整输出。
- T5 训练 (Span Corruption)：
  - Mask 策略：随机 Mask 掉 15% 的连续片段 (Span)（而非独立的 token），用哨兵符（如 `<extra_id_0>`）占位。
  - 目标：Decoder 自回归地生成被 Mask 掉的内容（如 `<extra_id_0> content <extra_id_1>`）。
  - 优势：
    - Seq2Seq 任务：在翻译、摘要、文本纠错等任务上，由于 Cross-Attention 的存在，Decoder 能时刻对齐源文本，生成内容的忠实度 (Faithfulness) 极高。
    - 多模态 (BLIP)：Vision Encoder 提取图像特征，Text Decoder 通过 Cross-Attention 注入图像信息，实现 Image Captioning。
- 对比 LLaVA (Decoder-only)：
  - T5/BLIP：层层交互。Decoder 每一层都在看 Encoder 的输出。
  - LLaVA：一次性注入。Vision Encoder 的特征仅在输入层（Input Embedding）拼接到 Text Embedding 前面，后续全是 Self-Attention。
  - 区别：T5 结构对细节控制更精准（适合 OCR/定位），LLaVA 结构泛化性更强（适合聊天/推理）。
##### 3. Decoder-only 架构
- 代表模型：GPT 系列, Llama, Qwen, DeepSeek
- 核心特点：Causal Attention (因果注意力)。每个 token 只能看到自己之前的 token（单向）。
- 训练 (CLM)：
  - Causal Language Modeling：预测下一个 token（Next Token Prediction）。
  - 优势：
    - Scaling Law：结构简单，并行效率高，最适合堆算力和数据。
    - 涌现能力：当参数量和数据量达到一定规模，展现出强大的通用推理和零样本能力。
    - 统一范式：通过 Prompt Engineering，可以用生成任务解决几乎所有 NLP 问题（分类、翻译、摘要等）。


> 💡 思考：Encoder-Decoder 家族内斗：BART vs T5 vs UL2
>
> *   BART (Bidirectional and Auto-Regressive Transformers)：
>     *   架构：最接近原始 Transformer 的标准结构（GPT 风格的 Decoder + BERT 风格的 Encoder）。
>     *   预训练目标：去噪自编码 (Denoising Autoencoder)。它不仅做 Span Mask，还做句子打乱 (Sentence Permutation)、Token 删除、文档旋转等多种噪声还原任务。
>     *   侧重：更像是一个“修复受损文本”的模型，因此在文本生成（如摘要、翻译）上表现极佳，但在理解任务（如分类）上略逊于 T5。
>
> *   T5 (Text-to-Text Transfer Transformer)：
>     *   架构：对 Transformer 做了微调（相对位置编码、LayerNorm 位置等），但核心还是 Encoder-Decoder。
>     *   预训练目标：Span Corruption (连续片段遮罩)。只做这一件事，但做得极好。
>     *   侧重：万物皆 Text-to-Text。它把分类、回归、翻译都统一成“输入文本 -> 输出文本”的形式，通用性极强。
>
> *   UL2 (Unifying Language Learning Paradigms)：
>     *   架构：基于 T5 改进，但核心是训练范式的统一。
>     *   预训练目标：Mixture of Denoisers (MoD)。它同时训练三种任务：
>         1.  R-Denoiser (Regular Span Corruption, 像 T5)：适合理解/短生成。
>         2.  S-Denoiser (Sequential Denoising, 像 GPT)：适合长生成/续写。
>         3.  X-Denoiser (Extreme Denoising)：适合极长上下文。
>     *   侧重：全能选手。它试图解决“T5 不擅长做 GPT 式续写，GPT 不擅长做 T5 式理解”的问题，是 Google 在 PaLM 之前的集大成者。
