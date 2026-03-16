cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

#set(CMAKE_CUDA_ARCHITECTURES "90a-real")

add_library(uniform1d_codegen_obj OBJECT
    uniform1d_launcher.cu
    uniform1d_split_bwd_u32_P22.cu
    uniform1d_split_bwd_u32_P777.cu
    uniform1d_split_bwd_u32_P1490.cu
    uniform1d_split_bwd_u32_P1554.cu
)

target_link_libraries(uniform1d_codegen_obj PUBLIC fasteq_deps)
target_include_directories(uniform1d_codegen_obj PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(uniform1d_codegen_obj PUBLIC cxx_std_17)

target_compile_options(uniform1d_codegen_obj PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    #$<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=--maxrregcount=128>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-warn-spills>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wno-unused-parameter>
    #$<$<BOOL:${DEBUG}>:$<$<COMPILE_LANGUAGE:CUDA>:-G;-g>>
)
