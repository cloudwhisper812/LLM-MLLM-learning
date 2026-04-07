### DPO

#### 算法核心：
- 数据要求：三元组数据(输入prompt，chosen answer, reject answer
- 模型结构：两个模型，reference model(sft后的模型，参数固定)，Policy Model（要训练的模型，参数初始化Reference Model 一样）
- 向前计算：分别用两个模型对好坏答案计算4组对数概率（Log Probabilities），其实就是每个位置token logit相乘，但是用log后是相加。
- loss理解：（1）policy model要比reference model 更大概率生成好答案 （2） 拉开margin，（模型对好回答的隐式打分） 减去 （模型对差回答的隐式打分）（3） $\beta$  是用来控制“偏离度”的超参（通常在 0.1 到 0.5 之间）。物理意义：它代表对 KL 散度的惩罚力度。 $\beta$  越大，模型越不敢偏离 Reference Model； $\beta$  越小，模型越放飞自我，疯狂迎合偏好数据，但也更容易过拟合（输出变成乱码或固定句式）。

#### 一些个人总结：
1. 


### PPO (Proximal Policy Optimization) 
需要4 个模型（假设都是 7B 大小）。它们的输入输出和状态如下：
- actor model：可训练，输入prompt，输出下一个token的logtis。
- reference model：冻结，sft阶段的模型副本。作为“锚点”。在 Actor 生成每一个 Token 时，Reference 也会算一下概率。如果 Actor 的概率分布偏离 Reference 太远，就会受到 KL 散度惩罚。这是为了防止模型“灾难性遗忘”或变成只会迎合奖励的复读机。
- Reward Model：冻结 (Frozen)（提前用人类偏好数据训练好的）。输入Prompt + Actor 生成的完整 Response。输出：一个标量（Scalar），比如 `+2.5` 或 `-1.2`，代表这句话的整体得分。
- Critic Model (价值模型 / Value Model)：可训练（通常用 Reward Model 初始化，然后跟着 Actor 一起训）。输入：Prompt + Actor 生成的部分 Response（即当前 State）。 输出：一个标量。它预测的是：“在当前这个状态下，未来我预期能拿到多少总奖励？”
