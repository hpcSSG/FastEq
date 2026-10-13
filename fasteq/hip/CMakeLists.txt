cmake_minimum_required(VERSION 3.21.3)

# Select the default architecture before HIP compiler initialization.
# Preserve explicit CMake and HIPARCHITECTURES environment settings.
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES
   AND "$ENV{HIPARCHITECTURES}" STREQUAL "")
  set(CMAKE_HIP_ARCHITECTURES gfx936 CACHE STRING "AMD GPU architectures")
endif()

set(FASTEQ_ROCM_ROOT "" CACHE PATH "ROCm or DTK installation root")
if(FASTEQ_ROCM_ROOT)
  list(PREPEND CMAKE_PREFIX_PATH "${FASTEQ_ROCM_ROOT}")
endif()

project(fasteq_hip LANGUAGES CXX HIP)

option(DEBUG "Enable debug macros and features" OFF)
option(FASTEQ_FAST_MATH "Enable HIP fast-math optimizations" ON)
option(FASTEQ_UNSAFE_FP_ATOMICS "Enable unsafe floating-point atomics" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

# Resolve PyTorch paths using the same Python interpreter as setuptools.
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c
    "import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent)"
  RESULT_VARIABLE _torch_result
  OUTPUT_VARIABLE _torch_root
  ERROR_VARIABLE _torch_error
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT _torch_result EQUAL 0)
  message(FATAL_ERROR "Cannot import PyTorch with ${Python3_EXECUTABLE}: ${_torch_error}")
endif()

if(NOT Torch_DIR)
  set(Torch_DIR "${_torch_root}/share/cmake/Torch" CACHE PATH "PyTorch CMake directory")
endif()
list(PREPEND CMAKE_PREFIX_PATH "${_torch_root}/share/cmake")

set(USE_KINETO OFF CACHE BOOL "" FORCE)
set(USE_PROFILER OFF CACHE BOOL "" FORCE)

find_package(hip CONFIG REQUIRED)
find_package(rocblas CONFIG QUIET)
find_package(hipblas CONFIG QUIET)
find_package(hipcub CONFIG QUIET)
find_package(Torch REQUIRED)

set(HIP_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/stc_fwd.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/stc_bwd.cpp"
)
foreach(_source IN LISTS HIP_SRCS)
  if(NOT EXISTS "${_source}")
    message(FATAL_ERROR "Missing source file: ${_source}")
  endif()
endforeach()

set_source_files_properties(${HIP_SRCS} PROPERTIES LANGUAGE HIP)
add_library(_hip MODULE ${HIP_SRCS})
target_include_directories(_hip PRIVATE ${TORCH_INCLUDE_DIRS})

# Apply PyTorch's ABI flags to HIP sources as well as C++ sources.
if(TORCH_CXX_FLAGS)
  separate_arguments(_torch_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
  target_compile_options(_hip PRIVATE ${_torch_flags})
endif()

target_compile_options(_hip PRIVATE
  $<$<COMPILE_LANGUAGE:HIP>:-Wno-return-type>
  $<$<AND:$<COMPILE_LANGUAGE:HIP>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-O3>
)
if(FASTEQ_FAST_MATH)
  target_compile_options(_hip PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-ffast-math>)
endif()
if(FASTEQ_UNSAFE_FP_ATOMICS)
  target_compile_options(_hip PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-munsafe-fp-atomics>)
endif()

set(FASTEQ_EXTENSION_NAME "_hip" CACHE STRING "Python extension module name")
set(FASTEQ_EXTENSION_SUFFIX "" CACHE STRING "Python extension ABI suffix")
if(NOT FASTEQ_EXTENSION_SUFFIX)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
      "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')"
    RESULT_VARIABLE _suffix_result
    OUTPUT_VARIABLE FASTEQ_EXTENSION_SUFFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT _suffix_result EQUAL 0)
    message(FATAL_ERROR "Cannot determine Python extension suffix")
  endif()
endif()

target_compile_definitions(_hip PRIVATE
  __HIP_PLATFORM_AMD__
  TORCH_EXTENSION_NAME=${FASTEQ_EXTENSION_NAME}
  TORCH_API_INCLUDE_EXTENSION_H
)
if(DEBUG)
  target_compile_definitions(_hip PRIVATE DEBUG=1)
endif()

# Match the pybind11 ABI settings used by the installed PyTorch build.
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c
    "import torch; print(';'.join(k + '=' + getattr(torch._C, '_PYBIND11_' + k) for k in ('COMPILER_TYPE', 'STDLIB', 'BUILD_ABI') if isinstance(getattr(torch._C, '_PYBIND11_' + k, None), str)))"
  RESULT_VARIABLE _pybind_result
  OUTPUT_VARIABLE _pybind_settings
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT _pybind_result EQUAL 0)
  message(FATAL_ERROR "Cannot determine PyTorch pybind11 ABI settings")
endif()
foreach(_setting IN LISTS _pybind_settings)
  string(REGEX MATCH "^([^=]+)=(.*)$" _match "${_setting}")
  if(_match)
    target_compile_definitions(_hip PRIVATE
      "PYBIND11_${CMAKE_MATCH_1}=\"${CMAKE_MATCH_2}\"")
  endif()
endforeach()

# Python tensor bindings may require symbols from libtorch_python.
find_library(TORCH_PYTHON_LIBRARY NAMES torch_python
  PATHS "${_torch_root}/lib" NO_DEFAULT_PATH REQUIRED)
target_link_libraries(_hip PRIVATE
  Python3::Module ${TORCH_LIBRARIES} "${TORCH_PYTHON_LIBRARY}" hip::device
)
foreach(_optional_target IN ITEMS roc::rocblas roc::hipblas hip::hipcub)
  if(TARGET ${_optional_target})
    target_link_libraries(_hip PRIVATE ${_optional_target})
  endif()
endforeach()

set_target_properties(_hip PROPERTIES
  PREFIX ""
  OUTPUT_NAME "${FASTEQ_EXTENSION_NAME}"
  SUFFIX "${FASTEQ_EXTENSION_SUFFIX}"
  POSITION_INDEPENDENT_CODE ON
)

message(STATUS "Python executable: ${Python3_EXECUTABLE}")
message(STATUS "Torch_DIR: ${Torch_DIR}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "HIP compiler: ${CMAKE_HIP_COMPILER}")
message(STATUS "HIP architectures: ${CMAKE_HIP_ARCHITECTURES}")
message(STATUS "Extension suffix: ${FASTEQ_EXTENSION_SUFFIX}")
