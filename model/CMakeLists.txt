find_package(OpenMP REQUIRED)
find_package(Armadillo REQUIRED)
find_package(glog REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(GTest REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(link_lib glog::glog GTest::gtest)
if (!WIN32)
    set(link_lib "${link_lib} pthread")
endif ()

set(link_math_lib ${ARMADILLO_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

aux_source_directory(./test DIR_TEST_ARMA)
aux_source_directory(${PROJECT_SOURCE_DIR}/engine/test ENGINE_TEST_SRC)
aux_source_directory(./src DIR_SOURCE_ARMA)
aux_source_directory(${PROJECT_SOURCE_DIR}/memory/src MEMORY_SRC)
aux_source_directory(${PROJECT_SOURCE_DIR}/utils/src UTILS_SRC)
aux_source_directory(${PROJECT_SOURCE_DIR}/tensor/src TENSOR_SRC)
aux_source_directory(${PROJECT_SOURCE_DIR}/kernels/src KERNELS_CPP_SRC)
aux_source_directory(${PROJECT_SOURCE_DIR}/layers/src LAYERS_SRC)
file(GLOB KERNELS_CU_SRC "${PROJECT_SOURCE_DIR}/kernels/src/*.cu")

add_executable(model_main model_main.cpp ${DIR_TEST_ARMA} ${ENGINE_TEST_SRC} ${DIR_SOURCE_ARMA} ${MEMORY_SRC} ${UTILS_SRC} ${TENSOR_SRC} ${KERNELS_CPP_SRC} ${KERNELS_CU_SRC} ${LAYERS_SRC})
# target_link_libraries(model_main ${link_lib} ${link_math_lib} OpenMP::OpenMP_CXX CUDA::cudart)
target_link_libraries(model_main ${link_lib} ${link_math_lib} OpenMP::OpenMP_CXX CUDA::cudart)


target_include_directories(model_main PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(model_main PUBLIC ${GTest_INCLUDE_DIR})
target_include_directories(model_main PUBLIC ${Armadillo_INCLUDE_DIR})
target_include_directories(model_main PUBLIC ./include)

target_include_directories(model_main PUBLIC ${PROJECT_SOURCE_DIR}/memory/include)
target_include_directories(model_main PUBLIC ${PROJECT_SOURCE_DIR}/utils/include)
target_include_directories(model_main PUBLIC ${PROJECT_SOURCE_DIR}/tensor/include)
target_include_directories(model_main PUBLIC ${PROJECT_SOURCE_DIR}/kernels/include)
target_include_directories(model_main PUBLIC ${PROJECT_SOURCE_DIR}/layers/include)
target_compile_definitions(model_main PRIVATE MYVLLM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

enable_testing()
add_test(NAME model_tests COMMAND model_main)
