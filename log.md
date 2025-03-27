# Log

## Week 5

学习并调研了大语言模型（LLM）相关的资源调度和推理优化方向的研究内容，总结以下几个关键点：
大规模模型推理的资源利用效率问题，包括如何更好地实现多节点间的协同调度。
推理过程中，任务的动态分配与负载均衡策略，以应对高并发场景下的性能瓶颈。
针对缓存和若干推理任务复用的高效机制探索（如通过存储热数据来减少重复计算）。
针对LLM的调度问题，结合调研结果，进行了以下分析和构思：
针对任务优先级不同的推理需求，初步提出了一种基于优先级动态分配的调度方案，以更高效分配关键任务的计算资源。
思考如何在模型分片（模型拆分到不同节点执行）场景下应用流水线并行和张量并行技术，优化跨设备的通信效率。
在实际环境中进行了部分实验探索：
对比了现有的静态调度方案和动态调度策略，收集推理效率的数据，将继续与团队讨论改进思路。
对多批次输入任务进行了初步的合并优化，尝试测试高并发场景下的吞吐量变化，以便验证调度方式的有效性。

## Week 6

本周我在本地跑了llama.cpp代码仓库，熟悉了仓库代码的逻辑。

阅读了几篇论文：

[1] InfLLM: Training-Free Long-Context Extrapolation for LLMs with an Efficient Context Memory.

[2] RetrievalAttention: Accelerating Long-Context LLM Inference via Vector Retrieval.

[3] PQCache: Product Quantization-based KVCache for Long Context LLM Inference.

[4] ∞Bench: Extending Long Context Evaluation Beyond 100K Tokens

[5] LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding

三篇论文的核心主要围绕long context llm inference时产生大量kv cache，针对llm context window不足或者出于推理时延需要对kv进行动态裁剪。其裁剪方式主要依赖各种近邻或者近似的近邻方法。

目前思路与下周方向：
- 复现三篇论文的方法，比较其在最新模型上的优劣，并分析其存在的问题（下周）
- 针对当前主流模型架构，设计更优的kv cache裁剪方案，在∞Bench和LongBench上去的更好的结果
- 将kv cache方案与OS和硬件相结合，实现hardware-aware kv cache裁剪
