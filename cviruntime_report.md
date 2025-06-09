### 代码仓库报告：`sophgo/cviruntime`


**主要功能：**
- 支持 CVITEK TPU 硬件（如 cv181x, cv182x, cv183x 等）。
- 提供共享内存和设备内存管理。
- 支持 TPU 上的命令执行和性能监控。
- 包含 Python 绑定以便快速开发和验证。
- 提供多个 SoC（System on Chip）平台的支持。

---

#### **代码结构**

1. **`src/` 目录**
   - **`common/`**
     - 包含共享逻辑，如内存分配、程序加载和运行时上下文管理。
     - 关键文件包括：
       - `program.cpp`: 负责 TPU 程序的加载和执行。
       - `shared_mem.cpp`: 管理共享内存的分配和回收。
       - `neuron.cpp`: 管理神经元相关的内存操作。
   - **`soc/`**
     - 包含针对不同 SoC（如 cv181x, cv182x）的具体实现。
     - 关键文件包括：
       - `tpu_pmu.cpp`: 用于 TPU 的电源管理和性能监控。
       - `cvi_device_mem.cpp`: 使用 ION 子系统管理设备内存。

2. **`doc/` 目录**
   - 包含开发手册和文档，比如 `cvitek_tpu_sdk_development_manual.md`，详细介绍了 TPU Runtime 的使用方法和 API。

3. **`samples/` 目录**
   - 提供了多个示例代码，用于展示如何使用 Runtime 开发 TPU 应用。
   - 示例包括分类器、BF16 模型等。

4. **`python/` 目录**
   - 提供 Python 绑定，使开发者可以快速使用 Runtime 的功能。

5. **`README.md` 文件**
   - 提供了项目的总体概述、依赖项和构建说明。

---

#### **如何向 TPU 发送命令（以 cv181x 为例）**

1. **准备命令缓冲区**
   TPU 的命令通常存储在缓冲区（`cmdbuf`）中。可以通过以下函数执行：
   ```cpp
   CVI_RT_RunCmdbufEx(_ctx, buf_mem, baseArray);
   ```
   此函数支持加密和非加密模式。

2. **数据传输到 TPU**
   数据通过以下步骤传输到 TPU：
   - 使用 `Neuron::toTpu()` 方法：
     ```cpp
     CVI_RT_MemFlush(_ctx, _gmem);
     CVI_RT_MemInvld(_ctx, _base_mem);
     ```
   - 此方法会确保数据从主机内存（Host Memory）刷新并加载到 TPU 内存。

3. **性能监控**
   在 `tpu_pmu.cpp` 文件中，TPU 的性能指标（如时钟速率、执行时间）通过以下逻辑计算：
   ```cpp
   percent_tdma = (double)u64TDMATotal / (double)bmnet_p_duration * 100;
   percent_tiu = (double)u64TIUTotal / (double)bmnet_p_duration * 100;
   ```

---

#### **内存类型及其关系**

1. **ION 内存**
   - 使用 Linux 的 ION 子系统进行内存分配。
   - 在文件 `cvi_device_mem.cpp` 中通过以下代码分配内存：
     ```cpp
     ret = ioctl(fd, ION_IOC_ALLOC, &alloc_data);
     ```

2. **共享内存**
   - 在 `shared_mem.cpp` 中，使用全局列表（`gSharedMemList`）管理共享内存块：
     ```cpp
     CVI_RT_MEM allocateSharedMemory(CVI_RT_HANDLE ctx, size_t size);
     deallocateSharedMemory(ctx, mem);
     ```

3. **主机内存**
   - 主机内存用于 CPU 端的操作，并在需要时映射到设备内存（TPU 内存）。

4. **关系总结**
   - **ION 内存** 作为底层分配器，为设备和共享内存提供物理内存。
   - **共享内存** 在多个进程间共享，基于 ION 内存实现。
   - **主机内存** 用于 CPU 端临时存储，并与设备内存交互。

### 模型推理流程分析（基于 `CVI_NN_Forward`）

以下是从模型初始化到推理执行的完整调用链和逻辑分析，基于 `sophgo/cviruntime` 中的代码。

---

### 1. **主要函数概述**

#### **`CVI_NN_Forward`**:
- 定义位置: `src/common/runtime.cpp`，第 271 行。
- 功能：执行模型的前向推理（阻塞模式），输入数据通过输入张量（`inputs`）传递，结果存储在输出张量（`outputs`）中。
- 核心逻辑：
  ```cpp
  CVI_RC CVI_NN_Forward(CVI_MODEL_HANDLE model, CVI_TENSOR inputs[], int32_t input_num, CVI_TENSOR outputs[], int output_num) {
      auto instance = (struct ModelInstance *)model;  // 解析模型实例
      if (instance->program->forward(inputs, input_num, outputs, output_num)) {
          return CVI_RC_SUCCESS;
      }
      return CVI_RC_FAILURE;
  }
  ```
  - 调用了 `instance->program->forward` 实现推理。

---

### 2. **推理前的准备工作**

#### **(1) 模型注册**
- 函数：`CVI_NN_RegisterModel`
- 作用：加载模型文件并初始化模型句柄。
- 示例代码：
  ```cpp
  const char *model_file = argv[1];
  CVI_MODEL_HANDLE model = nullptr;
  int ret = CVI_NN_RegisterModel(model_file, &model);
  if (ret != CVI_RC_SUCCESS) {
      printf("CVI_NN_RegisterModel failed, err %d\n", ret);
      exit(1);
  }
  printf("CVI_NN_RegisterModel succeeded\n");
  ```

#### **(2) 获取输入/输出张量**
- 函数：`CVI_NN_GetInputOutputTensors`
- 作用：获取模型的输入/输出张量信息。
- 示例代码：
  ```cpp
  CVI_TENSOR *input_tensors, *output_tensors;
  int32_t input_num, output_num;
  CVI_NN_GetInputOutputTensors(model, &input_tensors, &input_num, &output_tensors, &output_num);
  ```

---

### 3. **推理调用链**

#### **(1) `CVI_NN_Forward` 的调用**
`CVI_NN_Forward` 调用 `Program::forward` 执行推理：
- 定义位置：`src/common/program.cpp`，第 287 行。
- 核心逻辑：
  ```cpp
  bool Program::forward(CVI_TENSOR *inputs, int input_num, CVI_TENSOR *outputs, int output_num) {
      for (int i = 0; i < (int)out_tensors.size(); i++) {
          out_tensors[i]->store(outputs[i]);  // 将结果存储到输出张量
      }
      return true;  // 推理成功
  }
  ```

#### **(2) 数据传输到 TPU**
在推理过程中，数据会传输到 TPU：
- 使用 `Neuron::toTpu()`：
  ```cpp
  CVI_RT_MemFlush(_ctx, _gmem);
  CVI_RT_MemInvld(_ctx, _base_mem);
  ```

#### **(3) 异步推理支持**
- 函数：`CVI_NN_ForwardAsync`、`CVI_NN_ForwardWait`
- 作用：支持异步推理，先调用 `ForwardAsync` 提交任务，再用 `ForwardWait` 等待结果。

---

### 5. **关键调用链**

1. **模型加载**：
   - `CVI_NN_RegisterModel` -> 初始化模型句柄。
2. **输入/输出准备**：
   - `CVI_NN_GetInputOutputTensors` -> 获取输入/输出张量。
3. **推理执行**：
   - 调用 `CVI_NN_Forward` -> 内部调用 `Program::forward`。
4. **结果存储**：
   - 使用 `CVI_NN_TensorPtr` 获取结果。
