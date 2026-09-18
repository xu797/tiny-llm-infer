# Tensor Module

张量模块，基于memory模块的Buffer实现跨设备张量。

- Tensor：持有Buffer，管理张量维度、数据类型、设备信息
- 支持CPU/GPU张量创建、克隆、设备间数据拷贝
- 单元测试基于GTest，覆盖内存分配、张量克隆、CUDA数据读写
```
├── CMakeLists.txt              # tensor模块构建脚本，编译tensor库与GTest单元测试
├── include
│   ├── tensor.h                # Tensor张量类头文件；封装shape、data buffer、设备信息，提供clone、ptr、数据拷贝接口
│   └── utils.cuh               # CUDA测试工具函数声明
├── readme.md
├── src
│   ├── tensor.cpp              # Tensor类实现，依赖memory模块的Buffer与设备分配器
│   └── utils.cu                # CUDA测试辅助核函数，GPU侧数据填充
├── tensor_main.cpp
└── test
    ├── test_buffer.cpp         # Buffer内存分配/释放单元测试（复用memory模块buffer）
    └── test_tensor.cpp         # Tensor核心功能GTest：创建、clone、设备间拷贝、数据读写
````


