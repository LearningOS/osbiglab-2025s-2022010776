cmake_minimum_required(VERSION 3.12)

project(cvi_klm C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

if(NOT DEFINED TPU_SDK_PATH)
  message(FATAL_ERROR "Please set TPU_SDK_PATH to point to the TPU_SDK installation")
endif()
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type (default Debug)" FORCE)
endif()
include_directories(${TPU_SDK_PATH}/include)
link_directories(${TPU_SDK_PATH}/lib)

set(CVI_LIBS ${CVI_LIBS} cviruntime cvikernel)
#if(NOT CMAKE_CROSSCOMPILING)
#  set(CVI_LIBS ${CVI_LIBS} cvicmodel)
#endif()

set(EXTRA_LIBS ${EXTRA_LIBS} dl stdc++ pthread z)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED CONFIG)
#pybind11_add_module(cvi_klm naive/klm.cpp)
pybind11_add_module(cvi_klm KLM.cpp)

target_link_libraries(cvi_klm PRIVATE
    ${CVI_LIBS}
    ${EXTRA_LIBS})

install(TARGETS cvi_klm DESTINATION python)
