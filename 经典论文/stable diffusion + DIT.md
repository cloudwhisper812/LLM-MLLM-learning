##  Stable Diffusion

### 1. Latent Diffusion Model (LDM)
引入 **VAE (Variational Autoencoder)**：
* **Encoder**：将图像从 $512 x 512 x 3 压缩到 64 x 64 x 4 的隐空间（Latent Space）。
* **Decoder**：将去噪后的 Latent 还原回像素图像。

### 2. 训练流程
step1. text-image pair, 图片用encoder压缩成feature $z_0$  
step2. 对 $z_0$ 不停加噪声, $z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$  
step3. u-net 的输入是在t的噪声feature，参数t，以及文本的embedding(cross attenton引入)  
step4. u-net的输出和噪声算l2 loss，用u-net预测噪声。 真实训练中，对于一个图会sample一个t，直接加t次噪声，u-net只预测t的噪声。

inference的时候有多种采样器：DDPM ， DDIM ， Euler a （这部分还没看，后续看一下）

### 3. 模型结构
1. 参数t经过位置编码，然后全链接，然后加到每层的feature。
2. u-net 里面有self attention（用于建模图像内部的全局空间关系） + cross attention(k是图，q/v是文字，用来控制图)


## DiT (Diffusion Transformer)

### 1. 个人理解
这个不就是最直观vit用到stable diffusion的方式？把 Latent 切成 Patch 直接扔进vit。
甚至里面的控制方式adaLN-zero都是类似stylegan的adain？
不过感觉原文是class label用adaLN-zero，如果是长文本，应该还是cross attention更合适。
