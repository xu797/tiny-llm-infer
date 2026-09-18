# Status Module

状态码模块，统一定义项目内错误码与返回状态。

- Status：封装错误码与错误信息，用于函数返回值
- 提供成功/失败判断、错误信息获取接口
- 统一项目错误枚举，便于上层模块错误处理

````
.
├── CMakeLists.txt              # status模块构建脚本，编译status静态库
├── include
│   └── status.h                # Status类头文件，定义错误枚举与状态接口
└── src
    └── status.cpp              # Status类实现，错误信息构造与读取
````

