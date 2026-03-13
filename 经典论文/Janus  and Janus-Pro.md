## Janus / Janus-Pro: 视觉理解与生成的“强行大一统”

> 个人总结：这个idea很好，就像人一样从耳朵和眼睛接受不同信号，大脑统一处理。将图片生成和图片理解分别过不同的适合自己的encoder，然后交给LLM统一处理。



### 1. VQ (Vector Quantization) Tokenizer-VAE
输入一张原始图片。Encoder把图片压成一堆连续向量。量化层把这些向量强制映射到“词典”里最接近的 ID。Decoder 尝试把这些 ID对应的codebook里的vector拼到一起重建原始图片。codebook是和模型一起学习的可学习参数。loss是重建loss+词典对齐损失。

### 2. 之前的模型是怎么训练的？
A 路线：外挂式（如 LLaVA）—— 只管理解，不管生成
B 路线：强行一体化（如 Chameleon）—— 左右互搏。方法：用一个视觉编码器（比如把图片离散化成 Token）同时喂给 LLM 做输入和输出

### 3. Janus / Janus-Pro
> 理解任务：需要高层语义（High-level Semantics）。比如识别“猫”，模型不需要知道毛发的纹理，只需要知道猫的轮廓和特征。
> 生成任务：需要底层细节（Low-level Details）。比如画一只“猫”，模型必须精确控制每一个像素的纹理、光影。
> Janus 的核心贡献：它指出，不是模型的大脑（LLM）不行，而是眼睛（Encoder）不行。强行用一个视觉编码器去适配这两套完全相反的需求，会导致两边都“发育不良”。

#### 3.1 模型结构
理解端：用 SigLIP（类似 CLIP 但更猛），输出连续特征，专门负责看语义。
生成端：用 VQ Tokenizer（类似 VQ-VAE），把图片切块变成离散的视觉单词（Visual Tokens），专门负责搞像素。
计算过程：虽然输入端解耦了，但中间的“大脑”全是同一个 DeepSeek-LLM Transformer。大家最后都变成 Token 坐在同一个桌子上开会。

#### 3.2 训练流程：三步走
1. 多模态对齐（Alignment）：
- 只训练siglip后面的mlp，和vq token的embedding（随机初始化）层，扩充原本的词表。这里不需要codebook的embedding。
- 极其庞大但简单的图文对（Image-Text Pairs）。比如 LAION 或 CC3M 数据集。
- 理解端数据：图片 + 短标题（如：“一只白色的猫”）。生成端数据：短标题 + 图片。
- 理解任务：给定图片特征，算预测下一个文本 Token 的 CE Loss。生成任务：给定文本特征，算预测下一个视觉 Token (VQ ID) 的 CE Loss
   
2. 第二阶段：统一预训练（Unified Pre-training）
- MLP/embedding + 整个 LLM 基座 都在更新。视觉 Encoder 依然冻结
- 图文交错数据（Interleaved Data）：比如从网页抓取的一篇文章，里面一段字插一张图。这能让模型学会长上下文中的多模态关联。纯文本数据：必须掺杂大量的纯文本（代码、书籍），防止模型“灾难性遗忘”、
- Loss 怎么算：依然是 Next-token prediction 的 CE Loss
- 工程细节：为了节省计算量和防止概率空间冲突，Transformer 最后一层采用了双头设计，Text Head 预测文字，Vision Head 预测图像 ID

3. 第三阶段：指令微调（SFT）与 RL
- MLP/embedding + 整个 LLM 基座 都在更新。视觉 Encoder 依然冻结
- SFT 数据：几十万条极其高质量的人工构造指令。比如：“请仔细观察图片左下角，告诉我那个红色的物体是什么，并据此画一张它变大后的样子。”Loss 怎么算（SFT）：只对模型生成的回答（Response）部分计算 CE Loss，用户的提问（Prompt）部分不参与 Loss 计算（Mask 掉）。
- 第二阶段：全序列算 Loss（众生平等）。第三阶段：User Prompt 必须全部 Mask 掉（Loss 掩码）！

4. Janus-Pro 的大招：引入强化学习 (RL)
这是 Pro 版本能反超专职画图模型的关键。在 SFT 之后，Pro 版本不再用死板的 CE Loss，而是用 RL（类似 DeepSeek-R1 里的 GRPO 或 DPO 机制）。奖励信号（Reward）从哪来？规则奖励：格式对不对？有没有按要求生成图片？判别模型奖励（Reward Models）：外挂一个专门打分的模型。如果生成的图片符合文本描述（图文一致性高）且没有马赛克/畸变（美学评分高），就给一个高分（Positive Reward）。RL Loss 的本质：它不再逼迫模型去死记硬背某一个“标准答案”，而是鼓励模型去探索能拿到“高分”的概率分布。这就解释了为什么 Pro 画出来的图，在光影和质感上会有质的飞跃。

### 4. 对比stable diffusion优缺点 
- Janus / Janus-Pro：
  - 极致的语义遵循能力 (Semantic Precision)。SD 的“大脑”通常是 CLIP，对于复杂指令（如多个人物的空间关系、精确计数、大段文字描述）理解力很弱，经常漏细节。
  - 端到端的原生交互 (Native Agentic Flow)。用 SD 做机器人，你需要拼接流水线（ASR -> LLM 提取提示词 -> SD 生图）。Janus 是一个统一的 Agent，它可以在同一轮对话里既输出分析文本，又顺手甩出一张图，天然适合做“多模态个人助理”或“交互式教育机器人”。
- Stable Diffusion
  - 推理时延与速度 (Latency & Speed)：Janus 是自回归生成，必须一个 Token 一个 Token 往外蹦。Diffusion 模型生成是一整张 Latent Feature Map 并发去噪的。再配合 LCM (Latent Consistency Model) 或 Turbo 技术，SD 甚至可以 1 步到 4 步出图，实时渲染毫无压力。
  - 画质上限与高分辨率 (Aesthetics & High-Res Ceiling)：Janus 受限于 VQ Tokenizer 的离散特性，在极其精细的纹理边缘，放大看还是容易Artifacts。且由于 Context Window 的限制，生成超高分辨率图像的显存开销会呈平方级爆炸。SD 基于连续的 Latent 空间（VAE），天然适合处理连续的光影变化，目前在写实感、超高清（4K/8K）放大和极度精细的审美控制（如 ControlNet 控线）上，依然是不可撼动的霸主。
