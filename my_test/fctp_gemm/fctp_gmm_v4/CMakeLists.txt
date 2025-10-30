cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(multi_path_mm LANGUAGES CXX CUDA) # 同时支持C++和CUDA语言

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
find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

# 查找CUTLASS。CUTLASS是头文件库，通常只需包含其路径。
# 假设CUTLASS源码位于项目目录的'cutlass'子文件夹中，或已安装在系统路径
if(DEBUG)
    message(STATUS "Debug mode enabled, adding __FCTP_DEBUG__ macro and CUTLASS support...")
    find_path(CUTLASS_INCLUDE_DIR cutlass/cutlass.h
        HINTS
            ${PROJECT_SOURCE_DIR}/cutlass/include # 常见本地路径
            /app/work/cutlass/include              # 或通过环境变量CUTLASS_ROOT指定
        DOC "Path to CUTLASS include directory"
    )
    # 检查是否找到CUTLASS
    if(NOT CUTLASS_INCLUDE_DIR AND NOT CUTLASS_ROOT_DIR)
        message(WARNING "CUTLASS include directory not found. Please set CUTLASS_ROOT_DIR or place CUTLASS in the expected path.")
    else()
        get_filename_component(CUTLASS_ROOT_DIR ${CUTLASS_INCLUDE_DIR} DIRECTORY)
        message(STATUS "Found CUTLASS root: ${CUTLASS_ROOT_DIR}")
    endif()
endif()

# 设置包含目录
include_directories(cuda_impl)
include_directories(${TORCH_INSTALL_PREFIX}/include)

# 添加文件，指定CUDA源文件
add_library(${PROJECT_NAME} SHARED
    cuda_impl/gmm_v4.cu
)

# 为目标设置编译和链接标志
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr> # 允许CUDA constexpr放松限制
    # 其他CUDA特定编译选项...
    -O3
    --ptxas-options=-v
)

if(DEBUG)
    add_compile_options(-g)
    add_compile_definitions(
        __FCTP_DEBUG__
        TORCH_USE_CUDA_DSA
    )
    # 为目标设置CUTLASS包含目录
    target_include_directories(${PROJECT_NAME} PUBLIC
        ${CUTLASS_ROOT_DIR}/include
        ${CUTLASS_ROOT_DIR}/tools/util/include
    )
endif()

# 添加LibTorch的包含目录（通常find_package(Torch)已设置，但确保其传播）
target_include_directories(${PROJECT_NAME} PUBLIC ${TORCH_INCLUDE_DIRS})

# 链接LibTorch库和CUDA库到目标
target_link_libraries(${PROJECT_NAME} PUBLIC ${TORCH_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDA_LIBRARIES}) 

set_target_properties(${PROJECT_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 可选：显示重要变量以便验证
message(STATUS "Found Torch: ${TORCH_INCLUDE_DIRS}")
message(STATUS "CUDA Architectures targeted: ${CMAKE_CUDA_ARCHITECTURES}")

# cmake -DCMAKE_PREFIX_PATH=/opt/conda/lib/python3.11/site-packages/torch ..
# cmake -DCMAKE_PREFIX_PATH=/opt/conda/lib/python3.11/site-packages/torch -DDEBUG=ON ..