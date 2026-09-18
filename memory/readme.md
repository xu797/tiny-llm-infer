alloc.h

DeviceAllocator（基类）
├─ CPUDeviceAllocator
└─ CUDADeviceAllocator

业务层：Tensor / 推理输入输出
    ↓
Buffer层：封装一块内存，知道它是CPU还是CUDA内存，知道size、dtype
    ↓
Allocator层：只干一件事：allocate / free，返回void*裸指针
