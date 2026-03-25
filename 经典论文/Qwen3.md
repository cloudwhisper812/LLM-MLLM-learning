# Qwen3 

## Pre-training stage
Qwen3 的“三阶段预训练”（通识 -> 退火 -> 长文本）在框架上和 2.5差别不大。在预训练阶段重要有两个改进：
（1）VL 引擎强拆 PDF： 用 Qwen2.5-VL 直接去“读”海量的复杂 PDF。
（2）以前做 Data Mixture，往往是 Domain（领域）级别的，比如“分配 20% 算力给维基百科，30% 给代码库”。Qwen3在单条数据（Instance-level）上去打标。

pretrain分成3个stages：
（1）30T数据，sequence length 4096 
（2）用专业性知识STEM, coding, reasoning, synthetic data来训练模型.5T数据，sequence length 4096。 
（3）Max sequence length 32768. 不同长度混合。和2.5一样修改基频，并且在infer的时候用 DCA, YARN.

## Post training
Qwen3重要创新点是可以切换think mode。既可以做快速对话，也可以做复杂推理，不需要像2.5一样分成多个模型。

#### step 1. Long-CoT Cold start （sft）
- 目的是给模型注入推理的范式和格式，而不是在这压榨推理性能。
- 尽量减少训练数据和训练step。 Less is more, 如果sft训练太多，会让模型策略分布太尖锐，限制了后面强化学习的探索空间。

#### step 2. Reasoning RL （rl）
- 专门针对代码，数学进行 grpo。纯粹为了拔高模型的推理上限。仅仅170 step。
- 这里不是给模型注入新知识，而是unlock模型已经有的知识。
- 这里要精准测出模型的能力边界，给它喂刚好在边界上的数据，质量和难度匹配，远大于数据量。
- Entropy Bonus：防坍塌（防过早收敛）： 模型一旦蒙对一次拿到高分，极容易变成“复读机”，只会用一种死板的套路解题（即输出概率分布急剧变窄，发生策略坍塌）。逼探索（保持好奇心）： 强迫模型在解题时不能“太自信”，必须保留一定的随机性，去探索整个解题空间里的其他逻辑分支，从而真正突破智商上限。

#### step 3. thinking mode fusion （sft）
- Rejectiong Samping 数据
- 通过控制符 /think /no_think来控制。

#### step 4. general RL （rl）
- 在全领域进行RLHF。
- reward：（1）rule based reward （2） model based with reference answer： LLM as judger （3）model based without answer，基于人类偏好数据训练的景点RM。只用于哪些没有标准答案的开放性问题（写诗，闲聊，价值观）。（4）RAG场景抑制幻觉，设置特定reward，强迫模型引用context得到答案。

### 小模型（小于14b）的strong-to-weak distillation
- off-policy distillation： 收集teacher在think/no think两种模式的答案，让小模型SFT。teacher forcing。
- online-policy distillation：off的缺点是student一直看teacher的完美路径，实际推理时候一旦走错一步，面临从没见过状态，会彻底崩溃。所以提出了online，让student自己生成回答，把轨迹喂给teacher，计算每个token上的logits，然后和小模型kl。

