cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

#set(CMAKE_CUDA_ARCHITECTURES "90a-real")

add_library(uniform1d_codegen_obj OBJECT

    # SevenNet
    uniform1d_combine_u32_path22_fwd.cu
    uniform1d_combine_u32_path22_bwd.cu
    uniform1d_combine_u128_path16_fwd.cu
    uniform1d_combine_u128_path16_bwd.cu
    uniform1d_combine_u32_path777_fwd.cu
    uniform1d_split_regalloc_u32_path777_bwd.cu
    uniform1d_combine_u32_path1490_fwd.cu
    uniform1d_split_regalloc_u32_path1490_bwd.cu
    uniform1d_combine_u32_path1554_fwd.cu
    uniform1d_split_regalloc_u32_path1554_bwd.cu

    # MACE Large
    uniform1d_combine_u224_path16_bwd.cu
    uniform1d_combine_u224_path16_fwd.cu
    uniform1d_combine_u224_path215_bwd.cu
    uniform1d_combine_u224_path215_fwd.cu

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
