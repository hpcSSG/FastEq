# 内核专用的CMakeLists.txt
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# 创建CUDA内核静态库
add_library(equi_kernels STATIC
    equi_linear_fwd_f64.cu
    equi_linear_fwd_f32.cu
    impl.h
    ptx_inst.cuh
    device_cuda_helper.cuh
)

# 显式设置CUDA架构为sm90a（Hopper架构）
set_target_properties(equi_kernels PROPERTIES
    CUDA_ARCHITECTURES "90a-real;90a-virtual"  # 或者简化为 "90"
    CUDA_RUNTIME_LIBRARY Static
    POSITION_INDEPENDENT_CODE ON
)

# 设置CUDA编译选项，针对sm90a和TF32优化
target_compile_options(equi_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
    -g
    -O3
    -lcuda
    -Xcompiler=-Wall,-Wextra,-Wno-unused-parameter
    -std=c++17
    --gpu-architecture=sm_90a
    $<$<BOOL:${DEBUG}>:-lineinfo>
    >
)

# 包含目录
target_include_directories(equi_kernels PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# 针对TF32的优化选项
target_compile_options(equi_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
    -Xptxas --opt-level=3
    -Xptxas -v
    # 启用TF32张量核心
    --ptxas-options=-allow-expensive-optimizations=true
    >
)

# Timer
if(TIMER)
    message(STATUS "Test-time mode enabled, adding __TEST_TIME__ macro...")
    add_compile_definitions(__TEST_TIME__)
endif()
