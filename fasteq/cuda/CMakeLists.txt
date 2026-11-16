cmake_minimum_required(VERSION 3.18)
project(fasteq_cuda LANGUAGES CXX CUDA)

option(DEBUG "Enable debug macros and features" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Hopper H100
set(CMAKE_CUDA_ARCHITECTURES "90a-real")

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

# ---- CUTLASS (header-only) ---------------------------------------------------------------
set(CUTLASS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/cutlass"
    CACHE PATH "CUTLASS source root")
if(NOT EXISTS "${CUTLASS_ROOT}/include/cutlass/cutlass.h")
  message(STATUS "CUTLASS not found at ${CUTLASS_ROOT}; trying FetchContent")
  include(FetchContent)
  FetchContent_Declare(cutlass_source
    GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
    GIT_TAG 0b55a2f691d69981583568fd9eb69687b1f0de8a)
  FetchContent_GetProperties(cutlass_source)
  if(NOT cutlass_source_POPULATED)
    FetchContent_Populate(cutlass_source)
  endif()
  set(CUTLASS_ROOT "${cutlass_source_SOURCE_DIR}")
endif()
message(STATUS "CUTLASS root: ${CUTLASS_ROOT}")

# Build the eqlinear backends BEFORE find_package(Torch): Torch's CMake config rewrites
# CMAKE_CUDA_ARCHITECTURES to plain sm_90, which breaks the sm_90a TMA/WGMMA kernels.
add_subdirectory(src/eqlinear)

# Auto-detect Torch_DIR from the active Python
if(NOT Torch_DIR)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
      "import os, torch; print(os.path.join(torch.utils.cmake_prefix_path, 'Torch'))"
    OUTPUT_VARIABLE _Torch_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(EXISTS "${_Torch_DIR}/TorchConfig.cmake")
    set(Torch_DIR "${_Torch_DIR}" CACHE PATH "PyTorch CMake directory" FORCE)
  endif()
endif()

find_package(Torch REQUIRED)
find_library(EQLINEAR_TORCH_PYTHON_LIBRARY NAMES torch_python
  HINTS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED)
# Torch's config may have rewritten the architecture; restore the sm_90a targets.
set(CMAKE_CUDA_ARCHITECTURES "90a-real")
if(NOT TARGET CUDA::nvToolsExt)
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED GLOBAL)
endif()

# ---- Find CUDA runtime shipped with PyTorch (pip wheel) ----
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c
    "import os, torch; \
p=os.path.join(os.path.dirname(torch.__file__),'..','nvidia','cuda_runtime','lib'); \
print(os.path.abspath(p))"
  OUTPUT_VARIABLE TORCH_CUDA_RUNTIME_LIBDIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(EXISTS "${TORCH_CUDA_RUNTIME_LIBDIR}/libcudart.so.12")
  message(STATUS "Using PyTorch CUDA runtime libdir: ${TORCH_CUDA_RUNTIME_LIBDIR}")
else()
  message(WARNING "PyTorch CUDA runtime libdir not found: ${TORCH_CUDA_RUNTIME_LIBDIR} (libcudart.so.12 missing)")
endif()

add_library(fasteq_deps INTERFACE)
target_include_directories(fasteq_deps INTERFACE
  ${Python3_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(fasteq_deps INTERFACE
  Python3::Module
  ${TORCH_LIBRARIES}
)
target_compile_options(fasteq_deps INTERFACE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

include_directories(src)
# ---- sources (use absolute paths to avoid empty/invalid lists) ----
set(CUDA_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/fctp_fwd.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/fctp_bwd.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/eqlinear/dispatch/eqlinear_dispatch.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/eqlinear/dispatch/eqlinear_host_module.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/eqlinear/common/eqlinear_group_table.cpp"
)

add_subdirectory(src/atombit)

# Hard fail early if any source is missing (prevents "No SOURCES given")
foreach(_f IN LISTS CUDA_SRCS)
  if(NOT EXISTS "${_f}")
    message(FATAL_ERROR "Missing source file: ${_f}")
  endif()
endforeach()

add_library(_cuda MODULE
  ${CUDA_SRCS}
  $<TARGET_OBJECTS:atombit_obj>
)


# Prefer PyTorch-shipped CUDA runtime at runtime
if(EXISTS "${TORCH_CUDA_RUNTIME_LIBDIR}/libcudart.so.12")
  set_target_properties(_cuda PROPERTIES
    BUILD_RPATH "${TORCH_CUDA_RUNTIME_LIBDIR};$ORIGIN"
    INSTALL_RPATH "${TORCH_CUDA_RUNTIME_LIBDIR};$ORIGIN"
  )
else()
  set_target_properties(_cuda PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
  )
endif()

target_include_directories(_cuda PRIVATE
  ${Python3_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/src/eqlinear/common
  ${CMAKE_CURRENT_SOURCE_DIR}/src/atombit
)

target_compile_options(_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=--maxrregcount=128>
  #$<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-warn-spills>
)

target_link_libraries(_cuda PRIVATE
  Python3::Module
  ${TORCH_LIBRARIES}
  ${EQLINEAR_TORCH_PYTHON_LIBRARY}
  fasteq_eqlinear_tf32
  fasteq_eqlinear_fp32
  fasteq_eqlinear_fp64
)

target_compile_options(_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--gpu-architecture=sm_90a>
)

set_target_properties(_cuda PROPERTIES
  PREFIX ""
  SUFFIX ".so"
  POSITION_INDEPENDENT_CODE ON
)

message(STATUS "Torch_DIR: ${Torch_DIR}")
message(STATUS "Found Torch includes: ${TORCH_INCLUDE_DIRS}")
message(STATUS "CUDA arch: ${CMAKE_CUDA_ARCHITECTURES}")
