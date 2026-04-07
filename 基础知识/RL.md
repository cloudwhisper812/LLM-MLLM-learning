### DPO (Direct Preference Optimization)

#### 算法核心：
- 数据要求：三元组数据(输入prompt，chosen answer, reject answer
- 模型结构：两个模型，reference model(sft后的模型，参数固定)，Policy Model（要训练的模型，参数初始化Reference Model 一样）
- 向前计算：分别用两个模型对好坏答案计算4组对数概率（Log Probabilities），其实就是每个位置token logit相乘，但是用log后是相加。
- loss理解：（1）policy model要比reference model 更大概率生成好答案 （2） 拉开margin，（模型对好回答的隐式打分） 减去 （模型对差回答的隐式打分）（3） $\beta$  是用来控制“偏离度”的超参（通常在 0.1 到 0.5 之间）。物理意义：它代表对 KL 散度的惩罚力度。 $\beta$  越大，模型越不敢偏离 Reference Model； $\beta$  越小，模型越放飞自我，疯狂迎合偏好数据，但也更容易过拟合（输出变成乱码或固定句式）。

#### 一些个人总结：
1. 在实际业务中，构造具有高区分度（High Margin）的 Hard Negative 是提升 DPO 效果的关键。和positive非常接近，但是是negatvie的case，太简单的negative sft就能解决，上rl没有意义。
2. DPO的数据来源可以是人工，可以是AI 反馈（目前最主流的玩法），rule-based / Verifiable（效果通常不如ppo/grpop。DPO 只能在已有的两条路径里选好坏，而 PPO/GRPO 可以通过与环境的实时交互，主动探索出新的正确解法 ***这里需要理解一下***）。
3. 经典的dpo是off policy，现在经常做法是改成on policy了。
4. 对于 $\beta$ 的理解，当 $\beta$ 非常大时， $\pi_\theta$  只需要稍微偏离一点  $\pi_{ref}$  ，这个对数比值就会被放大很多。在算 Loss 时，模型发现：“哎哟，我稍微改变一点预测概率，Loss 的惩罚/奖励就剧烈波动。” 为了求稳，梯度会强迫 $\pi_\theta$ 死死抱住 $\pi_{ref}$ 的大腿，不敢轻易改变输出分布。这也就意味着 KL 惩罚极强。当 $\beta$ 极其微小，导致 $\log \frac{\pi_\theta}{\pi_{ref}}$ 这一项怎么变，算出来的数值都很小。模型发现：“既然偏离 Reference 模型没什么惩罚，那我就放飞自我吧！” 于是，模型会不顾一切地去把 $\pi_\theta(y_w)$ 推向 1，把 $\pi_\theta(y_l)$ 压向 0，完全抛弃了语言模型原本的语法和常识约束。这就会导致生成的回答变得极端、甚至崩坏成乱码。这也意味着 KL 惩罚失效。


### PPO (Proximal Policy Optimization) 
需要4 个模型（假设都是 7B 大小）。它们的输入输出和状态如下：
- actor model：可训练，输入prompt，输出下一个token的logtis。
- reference model：冻结，sft阶段的模型副本。作为“锚点”。在 Actor 生成每一个 Token 时，Reference 也会算一下概率。如果 Actor 的概率分布偏离 Reference 太远，就会受到 KL 散度惩罚。这是为了防止模型“灾难性遗忘”或变成只会迎合奖励的复读机。
- Reward Model：冻结 (Frozen)（提前用人类偏好数据训练好的）。输入Prompt + Actor 生成的完整 Response。输出：一个标量（Scalar），比如 `+2.5` 或 `-1.2`，代表这句话的整体得分。
- Critic Model (价值模型 / Value Model)：可训练（通常用 Reward Model 初始化，然后跟着 Actor 一起训）。输入：Prompt + Actor 生成的部分 Response（即当前 State）。 输出：一个标量。它预测的是：“在当前这个状态下，未来我预期能拿到多少总奖励？”
