cmake_minimum_required(VERSION 2.8.3)
project(yolo2)

option(CUDNN "Enable CuDNN" OFF)
option(FORTIFY "Fortify GCC compilation" OFF)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_FLAGS_RELEASE "-Ofast -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -DNDEBUG")
set(CUDA_NVCC_FLAGS_RELEASE --use_fast_math)
add_compile_options(-Wall -Wextra)
add_definitions(-DGPU=1)
if (${CUDNN})
  add_definitions(-DCUDNN=1)
endif()

if (FORTIFY)
  add_compile_options(-fstack-protector-all -Wstack-protector --param ssp-buffer-size=4 -D_FORTIFY_SOURCE=2 -O2 -g)
endif()

find_package(catkin REQUIRED COMPONENTS
  image_transport message_generation nodelet roscpp roslib roslint sensor_msgs std_msgs
)
find_package(CUDA REQUIRED)

roslint_cpp(include/darknet/yolo2.h src/darknet/yolo2.cpp src/yolo2_node.cpp src/yolo2_nodelet.cpp)
roslint_add_test()

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend and a run_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependencies might have been
##     pulled in transitively but can be declared for certainty nonetheless:
##     * add a build_depend tag for "message_generation"
##     * add a run_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

add_message_files(
  FILES
  Detection.msg
  ImageDetections.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  CATKIN_DEPENDS message_runtime
)

include_directories(
  SYSTEM ${catkin_INCLUDE_DIRS}
  SYSTEM ${CUDA_INCLUDE_DIRS}
  include
  darknet/src
)

cuda_add_library(yolo2
  darknet/src/activation_kernels.cu
  darknet/src/activation_layer.c
  darknet/src/activations.c
  darknet/src/avgpool_layer.c
  darknet/src/avgpool_layer_kernels.cu
  darknet/src/batchnorm_layer.c
  darknet/src/blas.c
  darknet/src/blas_kernels.cu
  darknet/src/box.c
  darknet/src/col2im.c
  darknet/src/col2im_kernels.cu
  darknet/src/connected_layer.c
  darknet/src/convolutional_kernels.cu
  darknet/src/convolutional_layer.c
  darknet/src/cost_layer.c
  darknet/src/crnn_layer.c
  darknet/src/crop_layer.c
  darknet/src/crop_layer_kernels.cu
  darknet/src/cuda.c
  darknet/src/data.c
  darknet/src/detection_layer.c
  darknet/src/dropout_layer.c
  darknet/src/dropout_layer_kernels.cu
  darknet/src/gemm.c
  darknet/src/gru_layer.c
  darknet/src/im2col.c
  darknet/src/im2col_kernels.cu
  darknet/src/image.c
  darknet/src/layer.c
  darknet/src/list.c
  darknet/src/local_layer.c
  darknet/src/matrix.c
  darknet/src/maxpool_layer.c
  darknet/src/maxpool_layer_kernels.cu
  darknet/src/network.c
  darknet/src/network_kernels.cu
  darknet/src/normalization_layer.c
  darknet/src/option_list.c
  darknet/src/parser.c
  darknet/src/region_layer.c
  darknet/src/reorg_layer.c
  darknet/src/rnn_layer.c
  darknet/src/route_layer.c
  darknet/src/shortcut_layer.c
  darknet/src/softmax_layer.c
  darknet/src/tree.c
  darknet/src/utils.c
)
cuda_add_cublas_to_target(yolo2)

target_link_libraries(yolo2
  ${CUDA_LIBRARIES}
  ${CUDA_curand_LIBRARY}
)

add_library(yolo2_nodelet
  src/yolo2_nodelet.cpp
  src/darknet/yolo2.cpp
)
add_dependencies(yolo2_nodelet yolo2_generate_messages_cpp)

target_link_libraries(yolo2_nodelet
  yolo2
  ${catkin_LIBRARIES}
)
