## ControlNet (Adding Conditional Control to Diffusion Models)

### 背景和核心思想
- 在 Stable Diffusion (SD) 刚开源的那段时间，虽然文生图的视觉效果很惊艳，但在实际的工业界落地中（比如设计工作流、视频特效）面临一个极其致命的问题：文本的控制力太弱了。 你无法用 Prompt 准确描述一个人的具体骨骼姿态、建筑的具体边缘透视或是精确的深度图。
- ControlNet 的核心思想非常像 ResNet 的残差连接，但它做到了极致的“插件化”。它绝对不碰（冻结） SD 模型原本的权重，而是把 SD 的 Encoder 复制一份出来作为“可训练副本”。在这个副本里接收新的空间条件（如边缘图、深度图），然后通过一种特殊的卷积结构，把提取到的条件特征“加”回到原模型对应的 Decoder 层中。

### 方法
> 我觉得这里可以看原文的图，非常清楚
- Freeze and Copy (冻结与复制)：将原始 SD 的 U-Net 锁定（Freeze Base Model）。把 U-Net 的 Encoder 和 Middle Block 复制一份，设为可训练（Trainable Copy）。 完美保留 Base Model 强大的自然图像先验知识
- Zero-Convolutions (零卷积)：权重和偏置在初始化时全部被强制设为 0。ControlNet 分支在训练初始阶段对 Base Model 的输出没有任何影响。随着训练的进行，梯度会流过零卷积，它的权重慢慢变得非零，开始逐步地、安全地将外部条件（如 Canny 边缘）注入到主干网络中，完全避免了初始随机噪声破坏主网络特征分布的问题。
- 小样本奇迹： 因为有零卷积的保护，即便是在单张消费级显卡上，只用几万张甚至几千张的数据集，也能在短时间内（几天）训练出一个特定条件（如手部骨骼控制）的 ControlNet，且绝不过拟合。

### 个人思考
1. 训练中没见过的condition图，testing的直接插入会不work。但是后来有一些工作试图解决这个问题。
2. 还是用一些conditon图来控制，如果是想用纯语言精准控制该怎么办？DiT, LLM控制SD。
