cmake_minimum_required(VERSION 3.18)
project(fasteq_cuda LANGUAGES CXX CUDA)

option(DEBUG "Enable debug macros and features" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Hopper H100
set(CMAKE_CUDA_ARCHITECTURES 90-real)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

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
find_package(CUDAToolkit REQUIRED)

# ---- sources (use absolute paths to avoid empty/invalid lists) ----
set(CUDA_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/fctp_fwd.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/fctp_bwd.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/stc_fwd.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/stc_bwd.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/equi_linear.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cwtp_fwd.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cwtp_bwd.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/mptp_fwd.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/mptp_bwd.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/spherical_harmonics.cu"
)

# Hard fail early if any source is missing (prevents "No SOURCES given")
foreach(_f IN LISTS CUDA_SRCS)
  if(NOT EXISTS "${_f}")
    message(FATAL_ERROR "Missing source file: ${_f}")
  endif()
endforeach()

add_library(_cuda MODULE ${CUDA_SRCS})

target_include_directories(_cuda PRIVATE
  ${Python3_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
)

target_compile_options(_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=--maxrregcount=128>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-warn-spills>
)

target_link_libraries(_cuda PRIVATE
  Python3::Module
  ${TORCH_LIBRARIES}
  CUDA::cudart
)

set_target_properties(_cuda PROPERTIES
  PREFIX ""
  SUFFIX ".so"
  POSITION_INDEPENDENT_CODE ON
)

message(STATUS "Torch_DIR: ${Torch_DIR}")
message(STATUS "Found Torch includes: ${TORCH_INCLUDE_DIRS}")
message(STATUS "CUDA arch: ${CMAKE_CUDA_ARCHITECTURES}")
