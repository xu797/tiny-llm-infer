# Memory Module
内存管理模块，提供跨设备统一内存抽象。
- 抽象基类 `DeviceAllocator`：定义分配、释放接口
- CPUAllocator：主机内存分配
- CUDAAllocator：GPU显存分配
- AllocatorFactory：工厂模式获取分配器单例
- Buffer：持有一块内存块，记录设备类型、字节大小，对外暴露内存指针

````
memory/
├── CMakeLists.txt              # memory模块构建脚本，编译allocator与buffer静态库
├── include
│   ├── alloc_factory.h         # 分配器工厂，统一获取CPU/CUDA分配器实例
│   ├── buffer.h                # Buffer内存缓冲区封装，管理一块设备内存（CPU/GPU）
│   ├── cpu_alloc.h             # CPU内存分配器，实现host侧malloc/free
│   ├── cuda_alloc.h            # CUDA内存分配器，实现device侧cudaMalloc/cudaFree
│   └── device_alloc.h          # 设备分配器基类，定义统一的分配/释放虚接口
├── readme.md
└── src
    ├── alloc_factory.cpp       # 分配器工厂实现，单例管理不同设备分配器
    ├── buffer.cpp              # Buffer类实现，内存持有、设备判断、指针获取
    ├── cpu_alloc.cpp           # CPU分配器实现，封装std::malloc / std::free
    ├── cuda_alloc.cpp          # CUDA分配器实现，封装cudaMalloc / cudaFree，CUDA错误检查
    └── device_alloc.cpp        # 设备分配器基类实现
````
