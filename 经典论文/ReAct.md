# ReAct: Synergizing Reasoning and Acting in Language Models
> 现在回头看感觉这都能发文章，纯prompt engineering。但是当时确实没有人想到这么简单的拆解？

### 核心本质：ReAct 到底是什么？
- Reason-Only / CoT: Let's think step by step。极度容易产生事实幻觉（Hallucination）。一旦脑子里的前置知识记错了，后面的逻辑推导全盘崩溃（错误级联）。
- Act-Only: 缺乏高层目标拆解能力和工作记忆，遇到 API 报错不知道换个思路，经常陷入死循环（瞎撞）。
- ReAct: 让大语言模型（LLM）与外部环境交互的提示词工程（Prompting）和系统控制范式。强迫模型在解决复杂任务时，严格按照 Thought (思考) -> Action (调用工具) -> Observation (环境真实反馈) 的死循环交替输出，实现了“脑子指导手，手拯救脑子”的完美闭环（Synergy）。

### 个人总结的一些论文值得记住的亮点
- ReAct 循环里的第一个 Thought（比如：“我需要先查 A，再查 B”）在物理意义上与后续的 Thought 完全不同。前者是全局任务规划（Planning），后者是局部进度追踪与异常处理（Execution Tracking）。这直接启发了后来的现代 Agent 框架（如 AutoGPT, LangChain），将系统强行拆解为独立 Planner 节点 + 独立 Executor 节点的模式。
- 论文中极其经典的两种“系统级切流策略”，已成为现今商业大模型的标配：原文里是试一个，不行再另一个。现在基本都是planner先判断，简单任务cot或者直接给出。复杂任务react+search internet。
- 微调 (Finetuning) 的降维打击红利：拿 3000 条高质量的 [Thought, Action, Obs] 轨迹数据去微调一个小模型，它跑 Agent 的成功率，竟然直接吊打靠 Prompting 驱动的千亿级巨兽。本质原因：微调 CoT 只是在教模型死记硬背（容易过拟合幻觉）；而微调 ReAct，是在教模型掌握一套通用技能（泛化能力）——即如何熟练使用外部工具、如何拆解逻辑、如何校验事实。这也是如今开源界狂卷 Tool-use / Function Calling SFT 数据的底层逻辑。


### 举个例子：电影《奥本海默》的导演，比电影《芭比》的导演大多少岁？

纯血大模型（无 ReAct）会怎么死？
如果没有 ReAct，大模型（比如纯 CoT）会试图在脑子里直接“背出”答案。如果它记错了 Greta Gerwig 的年龄，它就会一本正经地胡说八道：“奥本海默导演诺兰 53 岁，芭比导演 30 岁，大 23 岁。” —— 这就是幻觉（Hallucination），一旦中间某一步背错，全盘皆输。

ReAct 的真实运行日志（Step-by-Step）
现在，我们启动 ReAct Agent。请注意区分谁在说话：

【LLM 生成】：大模型输出的内容（包含思考和决定调用的动作）。

【外界环境返回】：你的 Python 代码拦截动作后，调用真实 API 返回的客观结果。

🔄 第 1 轮循环 (LLM 拆解目标)
【LLM 生成】 Thought 1: 我需要先找出《奥本海默》的导演是谁，以及《芭比》的导演是谁，然后再查他们的年龄。我先查《奥本海默》。

【LLM 生成】 Action 1: Search[奥本海默 导演]

(此时，大模型暂停生成，将控制权交还给你的 Python 脚本)

【外界环境返回】 Obs 1: 《奥本海默》是一部 2023 年的传记惊悚片，由克里斯托弗·诺兰（Christopher Nolan）编剧和执导。

🔄 第 2 轮循环 (LLM 根据真数据更新知识)
(Python 脚本把 Obs 1 拼接到 Prompt 里，再喂给大模型)

【LLM 生成】 Thought 2: 好的，《奥本海默》的导演是克里斯托弗·诺兰。现在我需要查他几岁了。

【LLM 生成】 Action 2: Search[克里斯托弗·诺兰 年龄]

【外界环境返回】 Obs 2: 克里斯托弗·诺兰出生于 1970 年 7 月 30 日（现年 55 岁）。

🔄 第 3 轮循环 (LLM 转移目标)
【LLM 生成】 Thought 3: 诺兰 55 岁。接下来我要找《芭比》的导演。

【LLM 生成】 Action 3: Search[芭比 电影 导演]

【外界环境返回】 Obs 3: 《芭比》是由格蕾塔·葛韦格（Greta Gerwig）执导的 2023 年奇幻喜剧片。

🔄 第 4 轮循环 (LLM 继续深挖)
【LLM 生成】 Thought 4: 导演是格蕾塔·葛韦格，我需要查她的年龄。

【LLM 生成】 Action 4: Search[格蕾塔·葛韦格 年龄]

【外界环境返回】 Obs 4: 格蕾塔·葛韦格出生于 1983 年 8 月 4 日（现年 42 岁）。

🔄 第 5 轮循环 (LLM 发现异常并调用新工具)
【LLM 生成】 Thought 5: 诺兰 55 岁，葛韦格 42 岁。题目问诺兰比她大多少岁。我需要计算 55 减去 42。虽然我可以直接算，但为了准确，我调用计算器。

【LLM 生成】 Action 5: Calculate[55 - 42]

【外界环境返回】 Obs 5: 13

✅ 第 6 轮循环 (LLM 整合信息，输出结果)
【LLM 生成】 Thought 6: 计算结果是 13。我现在拥有了回答用户问题的所有信息。

【LLM 生成】 Action 6: Finish[《奥本海默》的导演克里斯托弗·诺兰（55岁）比《芭比》的导演格蕾塔·葛韦格（42岁）大 13 岁。]

(Python 脚本拦截到 Finish，跳出 while 循环，把括号里的文字发给用户！)
