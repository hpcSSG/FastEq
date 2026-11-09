# eqlinear backends.  Each precision is a separate DSO with hidden visibility and a single
# exported C ABI entry point; the three lines' CUTLASS forks therefore never share a symbol table.
# This directory is added BEFORE find_package(Torch) in fasteq/cuda/CMakeLists.txt so Torch cannot
# downgrade sm_90a to sm_90.

find_package(CUDAToolkit REQUIRED)

set(EQLINEAR_COMMON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/common")

add_library(eqlinear_common_obj OBJECT
  "${EQLINEAR_COMMON_DIR}/eqlinear_group_table.cpp"
  "${EQLINEAR_COMMON_DIR}/packed_index_tables.cpp"
)
set_target_properties(eqlinear_common_obj PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)
target_include_directories(eqlinear_common_obj PRIVATE
  "${EQLINEAR_COMMON_DIR}"
  ${CUDAToolkit_INCLUDE_DIRS}
)

set(EQLINEAR_BACKEND_COMMON_OPTIONS
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr>
)

function(fasteq_eqlinear_backend name)
  set(options)
  set(oneValueArgs ARCH)
  set(multiValueArgs SOURCES DEFINES INCLUDES)
  cmake_parse_arguments(EQ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  add_library(fasteq_eqlinear_${name} SHARED
    $<TARGET_OBJECTS:eqlinear_common_obj>
    ${EQ_SOURCES}
  )
  set_target_properties(fasteq_eqlinear_${name} PROPERTIES
    CUDA_ARCHITECTURES "${EQ_ARCH}"
    CUDA_RUNTIME_LIBRARY Shared
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
  )
  target_compile_options(fasteq_eqlinear_${name} PRIVATE ${EQLINEAR_BACKEND_COMMON_OPTIONS})
  target_compile_definitions(fasteq_eqlinear_${name} PRIVATE ${EQ_DEFINES})
  target_include_directories(fasteq_eqlinear_${name} PRIVATE
    "${EQLINEAR_COMMON_DIR}"
    ${EQ_INCLUDES}
    ${CUDAToolkit_INCLUDE_DIRS}
    "${CUTLASS_ROOT}/include"
    "${CUTLASS_ROOT}/tools/util/include"
  )
endfunction()

# ---- TF32 -------------------------------------------------------------------------------------
set(EQLINEAR_TF32_DIR "${CMAKE_CURRENT_SOURCE_DIR}/backends/tf32")
fasteq_eqlinear_backend(tf32
  ARCH "90a-real;90a-virtual"
  SOURCES
    "${EQLINEAR_TF32_DIR}/api.cpp"
    "${EQLINEAR_TF32_DIR}/runner.cu"
    "${EQLINEAR_TF32_DIR}/bwd.cu"
    "${EQLINEAR_TF32_DIR}/transpose_kt.cu"
    "${EQLINEAR_TF32_DIR}/reduce_dw.cu"
    "${EQLINEAR_TF32_DIR}/reduce_fwd.cu"
    "${EQLINEAR_TF32_DIR}/grouped_gemm_tf32.cu"
    "${EQLINEAR_TF32_DIR}/sm90_grouped_gemm_tf32.cu"
    "${EQLINEAR_TF32_DIR}/split_k_plan.cu"
    "${EQLINEAR_TF32_DIR}/split_pack.cu"
    "${EQLINEAR_TF32_DIR}/prep_kernels.cu"
  DEFINES
    EQLIN_TB_M=128 EQLIN_TB_N=128 EQLIN_TB_K=32 EQLIN_TB_K3=16
    EQLIN_CLUSTER_N=1 EQLIN_CLUSTER_M=1 EQLIN_PINGPONG=0 EQLIN_STAGES=0 EQLIN_TMA_HINT=0
    CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED
  INCLUDES "${EQLINEAR_TF32_DIR}"
)

# ---- FP32 -------------------------------------------------------------------------------------
set(EQLINEAR_FP32_DIR "${CMAKE_CURRENT_SOURCE_DIR}/backends/fp32")
fasteq_eqlinear_backend(fp32
  ARCH "90a-real;90a-virtual"
  SOURCES
    "${EQLINEAR_FP32_DIR}/api.cpp"
    "${EQLINEAR_FP32_DIR}/runner.cu"
    "${EQLINEAR_FP32_DIR}/grouped_gemm_fp32.cu"
    "${EQLINEAR_FP32_DIR}/simt_gemm.cu"
    "${EQLINEAR_FP32_DIR}/tma_desc.cu"
    "${EQLINEAR_FP32_DIR}/reduce_partials.cu"
    "${EQLINEAR_FP32_DIR}/prep_kernels.cu"
    "${EQLINEAR_FP32_DIR}/bwd.cu"
    "${EQLINEAR_FP32_DIR}/prep_bwd.cu"
    "${EQLINEAR_FP32_DIR}/reduce_dw.cu"
  DEFINES
  INCLUDES "${EQLINEAR_FP32_DIR}"
)

# ---- FP64 -------------------------------------------------------------------------------------
set(EQLINEAR_FP64_DIR "${CMAKE_CURRENT_SOURCE_DIR}/backends/fp64")
fasteq_eqlinear_backend(fp64
  ARCH "90a-real"
  SOURCES
    "${EQLINEAR_FP64_DIR}/api.cpp"
    "${EQLINEAR_FP64_DIR}/runner.cu"
    "${EQLINEAR_FP64_DIR}/bwd.cu"
    "${EQLINEAR_FP64_DIR}/grouped_gemm_f64.cu"
    "${EQLINEAR_FP64_DIR}/direct_gemm_f64.cu"
    "${EQLINEAR_FP64_DIR}/direct_gemm_f64_tma.cu"
    "${EQLINEAR_FP64_DIR}/sm90_grouped_gemm_fp64.cu"
    "${EQLINEAR_FP64_DIR}/prep_kernels.cu"
  DEFINES
    F64_TB_M=64 F64_TB_N=64 F64_TB_K=16 F64_WARP_M=32 F64_WARP_N=32 F64_INS_K=4
    F64_STAGES=4 F64_EPI_COUNT=2
    F64M2_TB_M=128 F64M2_TB_N=64 F64M2_TB_K=16 F64M2_STAGES=4 F64M2_SMEM_ATOM=128
    F64M2_EPI_STAGES=2 F64M2_PINGPONG=0 EQLIN_TMA_HINT=0
  INCLUDES "${EQLINEAR_FP64_DIR}"
)
