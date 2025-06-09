# CVI KLM

这个目录主要包含在cvi芯片上运行KLM的runtime和相应的python代码。

## Environment

构建环境需要一个riscv64的Linux系统。需要安装python（推荐3.13版本）、cmake和gnu工具链。

## Build

建议可以build一个docker来做build，端侧build很慢。

```bash
pip install -r requirements.txt
mkdir build && cd build
cmake .. -DTPU_SDK_PATH=$(pwd)/../../cvitek_tpu_sdk -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=$(python -m pybind11 --cmakedir) ..
make -j 8
```

以上指令执行完以后会得到一个`cvi_klm.cpython-313-riscv64-linux-gnu.so`，这就是我们需要的runtime。

这个runtime暴露了许多接口，就不一一介绍了，我相信从名字上就能看出它们的作用。

## Running

接下来可以在python中使用这个runtime。有以下几个代码：
- `inference.py`：直接推理模型，使用runtime自带的kv管理，有相应的tracer代码，可以用于对拍验证runtime kv cache管理的正确性
- `inference_cpu.py`：在CPU上运行模型，可以用于验证端侧环境是否正确
- `inference_py.py`：使用python做kv cache管理，类似根目录的`inference.py`，有各种模式，有相应tracer代码，可以用于对拍验证runtime调用模型的正确性。
- `inference_serve.py`：用于为键盘暴露stdio服务接口，使用python做kv cache管理，方便debug
- `inference_serve_native.py`：用于为键盘暴露stdio服务接口，使用runtime做kv cache管理，适合生产环境

运行的时候需要带上环境变量`LD_LIBRARY_PATH`指向tpu sdk的lib目录。例如：

```bash
LD_LIBRARY_PATH=../../cvitek_tpu_sdk/lib python inference.py
```

这里必须要在root权限下运行，否则无法调用tpu。

