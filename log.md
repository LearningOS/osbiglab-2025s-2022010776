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

## Week 7

按照上周计划，尝试复现了三篇论文，发现其中三篇各自都有一些缺陷：
- inf llm：本身kvcache的裁剪是static的 从准确度角度的话就不太好，然后后续从os角度做优化的话空间也比较小。
- retrieval attention：本身工作并不扎实，他的方法其实很多指标上并没有超过他的baseline，但是他自己依然把自己结果加粗了，很有误导性；同时也缺了一些比较重要的baseline 比如quest这篇；然后还有效率对比，他只和不带kv cache的比较，这是一个显然不合格的baseline。
- PQCache：本身PQ环节比较繁琐，且并未考虑与硬件的结合。

经过一番搜寻，我认为这篇工作是一个更好的算法：quest[1] 这篇论文实验做的比较扎实，而且充分考虑了硬件配合，做了Page-wise的裁剪。这与vllm[2]这篇所提出的PagedAttention优化结合十分紧密。

另外，本周我与一位在做硬件的同学聊了聊，他正在试图在端侧设备上运行LLM。因此，本周我也花了一定时间在这个具体硬件上，最终总结发现端侧设备的挑战主要是：
- 内存较小，需要做offloading
- 算力不足，不能有太长的context
- 针对特定npu硬件，只支持静态算子，也就是输入形状必须确定，因此不支持变长sequence，必须要做kv cache裁剪，与现在我所在做的工作不谋而合

[1] Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference

[2] Efficient Memory Management for Large Language Model Serving with PagedAttention

未来计划：
- 将Quest原本代码在端侧设备上复现，并做可能的改进
- 结合具体硬件，实现高效kv裁剪
- 最终在端侧设备上实现人类速度实时高保真token预测

## Week 9

按照上周计划，本周主要的工作是解决硬件适配问题。我重新找了一个能兼容cvitek的tpu驱动的debian build。在这个build上，可以成功找到/dev/cvi-tpu0，如图所示：

![image](https://github.com/user-attachments/assets/bcd4f15f-050d-4886-bc8d-a7ecbe8b969e)

同时，在这个系统上，成功安装了riscv版的torch和numpy，如图所示：

![image](https://github.com/user-attachments/assets/88b25bb6-0e51-4814-a0a1-4243337d9d6c)

下周计划：
- cpp code base在硬件上debug一下，争取成功接通
- 训一下模型（这个视情况可能放到下下周）

## Week 10

按照上周计划，本周主要完成了代码在硬件上的适配，成功完成了TPU的初始化、内存分配、模型加载、模型调用。整体调用流程正在debug，还没完全完成。

下下周计划：
- 完成整体全流程跑通
- 完成模型对拍
- 完成profiling，对性能有初步估计
- 分析sdk+driver代码
- 模型训练

## Week 12

（生病了，跳过了一周）

## Week 13

按照上次计划，完成了模型推理全流程。可以通过输出logits对比。有了一个简单的计时。分析了sdk代码。

下周计划：
- 跑通完整demo


## Week 14

按照上周计划，本周主要完成了模型训练和上机推理调试。出于硬件问题，暂时还没能接上键盘。本周完成了自动补全输入的demo。

下周计划：
- 连上键盘
- 优化模型性能

## Week 15

按照上周计划，优化了模型性能，即将完成键盘连接，已经完成了80%。

答辩大纲：
- 背景介绍（相关技术、目标）
- 基本信息（芯片架构、片上资源等等）
- 整体架构
    - TPU
    - TPU Drive & Runtime
    - Model Runtime
    - KV Cache Management
    - User Interface
- 模型架构 & 训练
- 完整demo

下周计划：
- 完成键盘连接
- 准备答辩

