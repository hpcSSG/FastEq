cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

#set(CMAKE_CUDA_ARCHITECTURES "90a-real")

add_library(uniform1d_codegen_obj OBJECT
    uniform1d_codegen_path777_u32_fwd.cu
    uniform1d_codegen_path777_u32_bwd.cu
    uniform1d_codegen_path1490_u32_fwd.cu
    uniform1d_codegen_path1490_u32_bwd.cu
    uniform1d_codegen_path1554_u32_fwd.cu
    uniform1d_codegen_path1554_u32_bwd.cu
    uniform1d_codegen_path215_u224_fwd.cu
    uniform1d_codegen_path215_u224_bwd.cu
    uniform1d_launcher.cu
)

target_link_libraries(uniform1d_codegen_obj PUBLIC fasteq_deps)
target_include_directories(uniform1d_codegen_obj PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(uniform1d_codegen_obj PUBLIC cxx_std_17)

target_compile_options(uniform1d_codegen_obj PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wno-unused-parameter>
    $<$<BOOL:${DEBUG}>:$<$<COMPILE_LANGUAGE:CUDA>:-G;-g>>
)
