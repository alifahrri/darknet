cmake_minimum_required(VERSION 2.8.3)
project(darknet)

find_package(Threads)
find_package(CUDA)
find_package(OpenCV REQUIRED)

add_definitions(-DOPENCV)

include_directories(include)
include_directories(src)

FILE(GLOB DARKNET_FILES src/*.c)
FILE(GLOB EXAMPLES_FILES examples/*.c)

#remove this file
list(REMOVE_ITEM DARKNET_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/compare.c")
list(REMOVE_ITEM EXAMPLES_FILES "${CMAKE_CURRENT_SOURCE_DIR}/examples/darknet.c")
list(REMOVE_ITEM EXAMPLES_FILES "${CMAKE_CURRENT_SOURCE_DIR}/examples/rnn_vid.c")
list(REMOVE_ITEM EXAMPLES_FILES "${CMAKE_CURRENT_SOURCE_DIR}/examples/swag.c")
list(REMOVE_ITEM EXAMPLES_FILES "${CMAKE_CURRENT_SOURCE_DIR}/examples/voxel.c")
list(REMOVE_ITEM EXAMPLES_FILES "${CMAKE_CURRENT_SOURCE_DIR}/examples/dice.c")
list(REMOVE_ITEM EXAMPLES_FILES "${CMAKE_CURRENT_SOURCE_DIR}/examples/writing.c")

message("darkent files : ${DARKNET_FILES}")

add_definitions(-O4 -g)

if(CUDA_FOUND)
    link_directories(
        /usr/local/cuda/lib64
    )
    set(CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};
        -O3;
        -gencode arch=compute_30,code=sm_30
        -gencode arch=compute_35,code=sm_35;
        -gencode arch=compute_50,code=[sm_50,compute_50]
        -gencode arch=compute_52,code=[sm_52,compute_52]
        -gencode arch=compute_61,code=sm_61
        -gencode arch=compute_62,code=sm_62
    )
    message("CUDA_NVCC_FLAGS" ${CUDA_NVCC_FLAGS})
    add_definitions(-DGPU)
    FILE(GLOB DARKNET_CUDA_FILES src/*.cu)
    cuda_add_library(${PROJECT_NAME}lib SHARED ${DARKNET_FILES} ${DARKNET_CUDA_FILES})
    target_link_libraries(${PROJECT_NAME}lib 
        cuda
        curand
        cublas
        m
    )
else()
    list(APPEND LIBRARIES "m")
    add_library(${PROJECT_NAME}lib SHARED ${DARKNET_FILES})
endif()

include_directories(wrapper/include)
add_library(${PROJECT_NAME}_cpp_wrapper wrapper/src/yolowrapper.cpp yolo2_box.c)
target_link_libraries(${PROJECT_NAME}_cpp_wrapper ${PROJECT_NAME}lib)

add_executable(${PROJECT_NAME} examples/darknet.c)
add_library(example_lib ${EXAMPLES_FILES})
target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}lib ${OpenCV_LIBS} example_lib)