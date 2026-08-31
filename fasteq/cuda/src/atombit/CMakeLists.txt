cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

#set(CMAKE_CUDA_ARCHITECTURES "90a-real")

add_library(atombit_obj OBJECT
    fused_commutator.cu
    fused_bmm_sym.cu
)

target_link_libraries(atombit_obj PUBLIC fasteq_deps)
target_include_directories(atombit_obj PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(atombit_obj PUBLIC cxx_std_17)

target_compile_options(atombit_obj PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    #$<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wno-unused-parameter>
    $<$<BOOL:${DEBUG}>:$<$<COMPILE_LANGUAGE:CUDA>:-G;-g>>
)
