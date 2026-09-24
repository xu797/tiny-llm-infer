cmake_minimum_required(VERSION 3.20)
project(my_vllm LANGUAGES CXX CUDA)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

include_directories(${PROJECT_SOURCE_DIR})

add_subdirectory(tensor)
add_subdirectory(memory)
add_subdirectory(utils)
add_subdirectory(layers)
add_subdirectory(kernels)
add_subdirectory(model)
