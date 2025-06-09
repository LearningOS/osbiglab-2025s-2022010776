# Keyboard LM

本项目主要内容包括训练模型+在cvi芯片上部署模型。最终报告在Keyboard LM Final.pdf

## Overview

大致逻辑是：训练 -> 推理模型&验证效果 -> 转换模型为cvimodel格式 -> 端侧部署模型。

根目录下的文件负责干前三步，cvi_klm目录下的文件负责最后一步。

## Training

### Environment

训练环境需要一个预装好的python，并且推荐有GPU。

首先需要安装依赖：

```bash
pip install -r requirements.txt
```

可选安装：

```bash
pip install flash-attn --no-build-isolation
```

flash attn可以加快训练并减小内存训练时的内存峰值。

如果后续想把结果log到wandb，还需要执行

```bash
wandb login
wandb online
```

如果不想或者没有wandb，执行：

```bash
wandb offline
```

### Dataset

接下来预处理数据集：

```bash
mkdir data
python preprocess.py
```

这一步会自动下载Locutusque/UltraTextbooks数据集，并将其转换为训练所需的格式。其间需要连接huggingface.co，如网络连接不畅，可考虑预先下载数据集或者使用hf-mirror.com等镜像站。

### Start Training

接下来开始训练，如果你的环境有多个GPU并且有NCCL，可以进行多机训练，那么可以执行：

```bash
mkdir ckpt
torchrun --nnodes $(N_NODES) --nproc-per-node $(N_PROC_PER_NODE) train.py --config train.yaml
```

如果不满足以上条件，我没有试过，建议微调train.py，删除或注释掉ddp相关代码，然后再尝试。

训练配置在train.yaml里面。输出模型会保存到ckpt文件夹。

## Inference

### Main Inference Script

接下来尝试推理模型。

inference.py是主要的推理脚本，主要有以下参数：

```python
USE_KV_CACHE = True
USE_SEGMENTED = True
INTERACT = True
PREDICT = 1
```

作用分别是：
- `USE_KV_CACHE`：是否使用kv缓存，开启后可以加快推理速度。
- `USE_SEGMENTED`：是否使用分段模型，即segmenting_klm.py里定义的模型，用于验证segmenting_klm.py的正确性。
- `INTERACT`：是否进入交互模式，开启后可以输入一个Token推理一段。这个模式下可以实时显示预测准确率。
- `PREDICT`：每次推理多少个Token。

### Aligning Model Inference Modes

tracer.py用于对拍。基本运作方式是先用可靠的方式进行一次推理，中间根据配置tracer会把一些激活值记录下来，保存到`traced_tensors.pt`。然后再用不确定的方式运行一次，记录相同位置的激活值，然后就可以用tracer比较两次运行中各个激活值是否一样了。

各个函数用法如下：
- `running`：代表在某个设定下运行模型，主要用于记录这个设定的名字，以及指定是否开启激活值记录
- `executing`：代表正在跑某个模块，辅助记录位置信息
- `trace_tensor`：用于记录张量，给出张量名字和张量本身即可，可以接受numpy输入
- `compare`：开始对拍，脚本会根据记录下的tensor的名字对比两次或多次（没有测试过）running里，激活值是否一致，以及如果不一致，差了多少。

相应的有以下几个版本的模型：
- `inference.py`：基本版本，可以通过上一节的参数条件模式
- `inference_onnx.py`：使用onnx模型进行推理，主要用于验证onnx模型导出后的正确性
- `inference_with_prefill.py`：使用带prefill好的kv cache进行推理，可以验证prefill的正确性，以及方便与端侧设备上的版本进行对拍。

## Deployment

### Prefilling

首先，端侧不支持prefill，所以需要手动prefill一段前缀。在训练环境里执行：

```bash
mkdir prefilled
python prefill.py
```

这个的输出会存到prefilled里面，这个文件夹要传到端侧的cvi_klm目录下。

### Exporting ONNX

参照如下指令将模型转换为cvimodel。

现在训练环境下执行：

```bash
mkdir -p klm_models/segmented
python segmenting_klm.py
```

这一步会将训练好的模型重新划分成segment模型并且转换成onnx模式。对拍问题参照上一节。

### Compiling to CVI Model

拉取并运行tpu mlir的运行环境：

```bash
docker pull sophgo/tpuc_dev:latest
docker run --privileged --name cvi_compiler -v $PWD:/workspace -it sophgo/tpuc_dev:latest
```

接下来在tpu_mlir的环境下执行：

```bash
pip install tpu_mlir
cd /workspace
make build_segments
```

这一步会将模型转换为cvimodel格式。将cvi_model都传到端侧设备上。

接下来的步骤参见cvi_klm目录的指引。


