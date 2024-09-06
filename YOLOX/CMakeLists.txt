cmake_minimum_required(VERSION 3.8)

project(YOLOX DESCRIPTION "YOLOX inference demo")

set(TENSORRT_PATH /opt/nvidia/TensorRT-10.3.0.26)
set(TENSORRT_ROOT /opt/nvidia/TensorRT)
#set(TENSORRT_INCLUDE_DIR ${TENSORRT_ROOT/include})

include(
	../TensorRT.cmake
)

include_directories(
	../include
)

add_executable(YOLOX
	main.cpp
    YOLOX.cpp
	../src/TensorRT.cpp
	../src/Object.cpp
)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PkgConfig)
set(CUDA_LIB_PATH /usr/local/cuda/lib64)


target_include_directories(YOLOX PUBLIC
	${CUDA_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
	${TENSORRT_INCLUDE_DIRS}
)

target_link_libraries(YOLOX
	${CUDA_LIB_PATH}/libcublas.so
	${CUDA_LIB_PATH}/libcudart.so
	${CUDA_LIBRARIES}
	${OpenCV_LIBRARIES}
	${TENSORRT_LIBRARIES}
)
