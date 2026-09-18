find_package(OpenMP REQUIRED)
# find_package(Armadillo REQUIRED)
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
aux_source_directory(./src DIR_SOURCE_ARMA)
aux_source_directory(${PROJECT_SOURCE_DIR}/memory/src MEMORY_SRC)
aux_source_directory(${PROJECT_SOURCE_DIR}/utils/src UTILS_SRC)

add_executable(tensor_main tensor_main.cpp ${DIR_TEST_ARMA} ${DIR_SOURCE_ARMA} ${MEMORY_SRC} ${UTILS_SRC})
target_link_libraries(tensor_main ${link_lib} ${link_math_lib} OpenMP::OpenMP_CXX CUDA::cudart)

target_include_directories(tensor_main PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(tensor_main PUBLIC ${GTest_INCLUDE_DIR})
target_include_directories(tensor_main PUBLIC ${Armadillo_INCLUDE_DIR})
target_include_directories(tensor_main PUBLIC ./include)

target_include_directories(tensor_main PUBLIC ${PROJECT_SOURCE_DIR}/memory/include)
target_include_directories(tensor_main PUBLIC ${PROJECT_SOURCE_DIR}/utils/include)

enable_testing()