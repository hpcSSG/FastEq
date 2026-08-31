cmake_minimum_required(VERSION 3.18)
project(fasteq_hip LANGUAGES CXX HIP)

option(DEBUG "Enable debug macros and features" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

# AMD GPU architectures (MI100: gfx908, MI200: gfx90a, BW100: gfx936)
set(CMAKE_HIP_ARCHITECTURES gfx936 )

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

set(USE_KINETO OFF CACHE BOOL "" FORCE)
set(USE_PROFILER OFF CACHE BOOL "" FORCE)

find_package(hip REQUIRED)
find_package(rocblas QUIET)
find_package(hipblas QUIET)
find_package(hipcub QUIET)
find_package(Torch REQUIRED)

# ---- sources (use absolute paths to avoid empty/invalid lists) ----
set(HIP_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/stc_fwd.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/stc_bwd.cpp"
)

# Hard fail early if any source is missing (prevents "No SOURCES given")
foreach(_f IN LISTS HIP_SRCS)
  if(NOT EXISTS "${_f}")
    message(FATAL_ERROR "Missing source file: ${_f}")
  endif()
endforeach()

add_library(_hip MODULE ${HIP_SRCS})

set_source_files_properties(${HIP_SRCS} PROPERTIES LANGUAGE HIP)

target_include_directories(_hip PRIVATE
  ${Python3_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
)

add_compile_options(-Wno-return-type)

target_compile_options(_hip PRIVATE
  $<$<COMPILE_LANGUAGE:HIP>:-O3>
  $<$<COMPILE_LANGUAGE:HIP>:-ffast-math>
  $<$<COMPILE_LANGUAGE:HIP>:-munsafe-fp-atomics>
)

target_compile_definitions(_hip PRIVATE
  __HIP_PLATFORM_AMD__
)

target_link_libraries(_hip PRIVATE
  Python3::Module
  ${TORCH_LIBRARIES}
  hip::device
)

if(rocblas_FOUND)
  target_link_libraries(_hip PRIVATE roc::rocblas)
endif()

if(hipblas_FOUND)
  target_link_libraries(_hip PRIVATE roc::hipblas)
endif()

if(hipcub_FOUND)
  target_link_libraries(_hip PRIVATE hip::hipcub)
endif()

set_target_properties(_hip PROPERTIES
  PREFIX ""
  SUFFIX ".so"
  POSITION_INDEPENDENT_CODE ON
)

message(STATUS "Torch_DIR: ${Torch_DIR}")
message(STATUS "Found Torch includes: ${TORCH_INCLUDE_DIRS}")
message(STATUS "HIP arch: ${CMAKE_HIP_ARCHITECTURES}")