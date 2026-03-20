## positional embedding
> Transformer对位置不敏感（Self-Attention是集合运算，打乱顺序结果不变），所以必须手动注入位置信息。

### 1. 绝对位置编码
#### 1.1 Sinusoidal PE （Attention Is All You Need 原文）
- sin，cos function，变量是pos index，以及每个token中单个数值的index
- 可以外推(Extrapolation)，但是效果一般
#### 1.2 Learnable PE （BERT, GPT-2, ViT）
- 在固定长度任务上表现好，比如原始vit的输入图像尺寸固定。完全没法外推

### 2. 相对位置编码 Relative PE 
#### 2.1 RoPE (Rotary Positional Embedding) 现在的主流
* **核心思想**：通过在复平面上乘以一个与位置 $`m`$ 相关的旋转矩阵，将绝对位置信息注入到 Query 和 Key 中。使得它们的内积（Attention Score）自然而然地只与它们的相对位置 $`(m - n)`$ 相关。

* **算法重点**：
  * RoPE 将 $`d`$ 维特征向量两两一对，划分为 $`d/2`$ 个二维子空间。每个子空间使用不同的频率进行旋转。$`\theta_i = b^{-2i/d}`$ 其中 $`b`$ 就是基频（Base Frequency），默认值通常是 $`10000`$。$`i`$ 是特征维度的索引（从 $`0`$ 到 $`d/2-1`$）。
  
* **每个位置特征向量的旋转矩阵变化公式：**

```math
\begin{pmatrix}
\hat{q}_{m,2i} \\
\hat{q}_{m,2i+1}
\end{pmatrix}
=
\begin{pmatrix}
\cos(m\theta_i) & -\sin(m\theta_i) \\
\sin(m\theta_i) & \cos(m\theta_i)
\end{pmatrix}
\begin{pmatrix}
q_{m,2i} \\
q_{m,2i+1}
\end{pmatrix}
```

  * 高频区（$`i`$ 较小，靠前的维度）：$`\theta_i`$ 很大。位置 $`m`$ 每增加 1，向量在复平面上旋转的角度就极大（转得飞快）。物理意义：对相对距离极度敏感。只要 $`(m-n)`$ 稍微变一点，内积就剧烈震荡。这部分维度负责捕捉极其局部的微观语义（比如主谓宾搭配、紧邻词）。
  * 低频区（$`i`$ 较大，靠后的维度）：$`\theta_i`$ 极小（接近于 0）。位置 $`m`$ 增加几千，向量可能才转了一小圈。物理意义：对相对距离很不敏感。相隔 10 个 Token 和相隔 15 个 Token，内积几乎没变化。这部分维度负责捕捉全局的、宏观的上下文语义（比如整篇文章的主题）。
  * 乘法注入：相比于加法（Add），旋转（Multiply）保留了向量的模长，信息损失更少。

### 训练期 ABF (Adjusted Base Frequency) 与 Qwen 的策略

  * 为了让模型支持长文本，最暴力的思路是直接用 32K 的文本从头训，但这太费算力。先短后长，ABF（调整基频，也叫 RoPE Base Scaling）提供了一种极其优雅的数学平滑过渡。
  * ABF 的核心操作：放大基频 $`b`$。将基频 $`b`$ 从默认的 $`10000`$ 放大到 $`1,000,000`$（甚至 Qwen-Turbo 用的 $`10,000,000`$）。看回公式：$`\theta_i' = (b')^{-2i/d}`$。当 $`b'`$ 变大时，整体的 $`\theta_i`$ 会被压缩变小。
  * 物理直觉：ABF 让模型产生了一种“错觉”——它把极其遥远的 Token，在视觉上拉近到了自己熟悉的训练视野范围内。

- 推理期：YaRN (Yet another RoPE extensioN) 技术细节
  * 这段后续再看吧
- 3D RoPE（后面看一下）
#### 2.2 Alibi
- 直接对attention map上的attention score 减去一个线性偏置。相对距离。距离越远，减得越多。
- 无限外推。适合超长序列
- 为了让模型能关注不同范围的上下文，ALiBi 给每个 Head 分配了不同的斜率 m（乘法力度，线性偏置的权重。）
