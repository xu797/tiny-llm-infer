add_library(infer_engine STATIC
    src/sequence.cpp
    src/block_manager.cpp
    src/scheduler.cpp
    src/llm_engine.cpp
    src/qwen3_runner.cpp
)

target_compile_features(infer_engine PUBLIC cxx_std_17)
target_include_directories(infer_engine PUBLIC
    ${PROJECT_SOURCE_DIR}/engine/include
    ${PROJECT_SOURCE_DIR}/model/include
    ${PROJECT_SOURCE_DIR}/layers/include
    ${PROJECT_SOURCE_DIR}/tensor/include
    ${PROJECT_SOURCE_DIR}/utils/include
    ${PROJECT_SOURCE_DIR}/memory/include
    ${PROJECT_SOURCE_DIR}/kernels/include
)

find_package(glog REQUIRED)
target_link_libraries(infer_engine PUBLIC glog::glog CUDA::cudart CUDA::cublas)
find_package(CUDAToolkit REQUIRED)

