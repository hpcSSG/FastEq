cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(CMAKE_CUDA_ARCHITECTURES "90a-real")

add_library(equi_kernels_obj OBJECT
    equi_linear_fwd_f64.cu
    equi_linear_fwd_f32.cu
    equi_linear_bwd_f32.cu
)

target_include_directories(equi_kernels_obj PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(equi_kernels_obj PUBLIC cxx_std_17)

target_compile_options(equi_kernels_obj PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    #$<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wno-unused-parameter>
    #$<$<BOOL:${DEBUG}>:$<$<COMPILE_LANGUAGE:CUDA>:-G;-g>>
)

# Timer
if(TIMER)
    target_compile_definitions(equi_kernels_obj PRIVATE __TEST_TIME__)
endif()
