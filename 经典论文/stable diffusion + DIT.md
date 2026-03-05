##  Stable Diffusion

### 1. Latent Diffusion Model (LDM)
引入 **VAE (Variational Autoencoder)**：
* **Encoder**：将图像从 $512 x 512 x 3 压缩到 64 x 64 x 4 的隐空间（Latent Space）。
* **Decoder**：将去噪后的 Latent 还原回像素图像。

### 2. 训练流程
step1. text-image pair, 图片用encoder压缩成feature $z_0$  
step2. 对$z_0$不停加噪声, $z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$  
step3. 


U-Net 与 Cross-Attention
* **U-Net 角色**：作为噪声预测器（Noise Predictor），输入带噪 Latent $z_t$ 和步数 $t$，输出预测噪声 $\epsilon$。
* **Attention 机制**：
    * **Self-Attention**：用于建模图像内部的全局空间关系。
    * **Cross-Attention**：将文本 Embedding（由 CLIP 提取）注入 U-Net，实现 Text-to-Image 的控制。
* **公式理解**：$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d}})V$，其中 $Q$ 来自图像特征，$K, V$ 来自文本特征。



### 3. 采样 (Sampling / Scheduler)
* **本质**：求解随机微分方程（SDE/ODE）的过程。
* **常用采样器**：
    * **DDPM**：随机性强，步数多（1000步）。
    * **DDIM**：确定性采样，支持更少步数（20-50步），且支持 Inversion（反推噪声）。
    * **DPM++ / Euler a**：目前工业界兼顾速度与质量的首选。

---

## Ⅱ. DiT (Diffusion Transformer)

### 1. 从 U-Net 到 Transformer 的演进
* **背景**：Sora 和 SD3 的成功证明了 Scaling Law 在生成领域依然成立。U-Net 虽好，但参数量扩展（Scaling）不如 Transformer 顺滑。
* **DiT 核心结构**：
    * **Patchify**：效仿 ViT，将 $64 \times 64$ 的 Latent 切成一个个 Patch。
    * **DiT Block**：取代卷积层，使用标准 Transformer Block 处理 Patch。
    * **Adaptive Layer Norm (adaLN)**：将时间步 $t$ 和条件标签 $c$ 注入模型的核心手段。



### 2. Scaling Law 与视频生成
* **DiT 的优势**：模型深度和宽度更容易增加，更适合处理 **视频序列 (3D Attention)**。
* **应用案例**：Sora、SD3、Flux。

---

## Ⅲ. 算法工程师视角：TikTok 业务思考
* **内容安全**：如何利用 SD 的 Inpainting（局部重绘）能力进行违规内容遮盖或修复？
* **多模态增强**：SigLIP 2 能否作为更好的 Vision Tower 取代 CLIP，提供更精准的 Prompt 引导？

---
**Next Step**: [ ] 实现一个简单的 DDIM Sampler 代码 | [ ] 比较 DiT 与 U-Net 的 FLOPs 差异