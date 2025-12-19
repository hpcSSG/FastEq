cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(fasteq LANGUAGES CXX CUDA) # 同时支持C++和CUDA语言

# 定义一个名为 DEBUG 的选项，默认值为 OFF，描述信息可选
option(DEBUG "Enable debug macros and features" OFF)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 设置CUDA标准
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 重要：设置目标GPU架构（根据您的GPU调整，例如RTX 30系列常用sm_86）
# 可以通过命令行 nvidia-smi --query-gpu=compute_cap --format=csv 查询计算能力
set(CMAKE_CUDA_ARCHITECTURES "90-real") # 例如75代表计算能力7.5（Turing架构RTX 20系列）

# 查找必需的LibTorch包
find_package(Python3 REQUIRED COMPONENTS Development)
find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)



# 设置包含目录
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${TORCH_INSTALL_PREFIX}/include)

# 添加文件，指定CUDA源文件
add_library(${PROJECT_NAME} SHARED
    #fctp_fused2_fwd.cu
    #fctp_fused3_fwd.cu
    #fctp_fused3_bwd.cu
    fctp_fused_path_fwd.cu
    fctp_fused_path_bwd.cu
    #fctp_spmm_fwd.cu
    #fctp_spmm_bwd.cu
    stc_fwd.cu
    stc_bwd.cu
    equi_linear.cu
    cwtp_fwd.cu
    cwtp_bwd.cu
    fused_message_passing_fwd.cu
    fused_message_passing_bwd.cu
    spherical_harmonics.cu
)

# 为目标设置编译和链接标志
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr> # 允许CUDA constexpr放松限制
    # 其他CUDA特定编译选项...
    --std=c++17
    -O3
    -Xptxas --maxrregcount=128
    --use_fast_math
    --ptxas-options=-v
)

# 添加LibTorch的包含目录（通常find_package(Torch)已设置，但确保其传播）
target_include_directories(${PROJECT_NAME} PUBLIC ${Python3_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})


# 链接LibTorch库到目标
target_link_libraries(${PROJECT_NAME} PUBLIC ${Python3_LIBRARIES} ${TORCH_LIBRARIES})

set_target_properties(${PROJECT_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 可选：显示重要变量以便验证
message(STATUS "Found Torch: ${TORCH_INCLUDE_DIRS}")
message(STATUS "CUDA Architectures targeted: ${CMAKE_CUDA_ARCHITECTURES}")

# cmake -DCMAKE_PREFIX_PATH= /public-data/malixian/conda/fasteq/lib/python3.10/site-packages/torch ..
