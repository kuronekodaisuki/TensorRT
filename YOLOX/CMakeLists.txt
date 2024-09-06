cmake_minimum_required(VERSION 3.8)

project(YOLOX DESCRIPTION "YOLOX inference demo")

set(TENSORRT_PATH /opt/nvidia/TensorRT)

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
#find_package(TENSORRT REQUIRED)
#find_package(PkgConfig)


target_include_directories(YOLOX PUBLIC
	${CUDA_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
	${TENSORRT_INCLUDE_DIRS}
)

target_link_libraries(YOLOX
	${CUDA_LIBRARIES}
	${OpenCV_LIBRARIES}
	${TENSORRT_LIBRARIES}
)
