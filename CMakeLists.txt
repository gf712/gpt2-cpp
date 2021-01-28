cmake_minimum_required(VERSION 3.1)
project(GPT2)

set(CMAKE_CXX_STANDARD 17)

find_path(ORT_INCLUDE_DIR
            NAMES "onnxruntime_cxx_api.h"
            HINTS ENV "HOME"
            PATH_SUFFIXES onnxruntime/include/onnxruntime/core/session
            REQUIRED)

find_library(ORT_LIBRARY
             NAMES "libonnxruntime.so"
             HINTS ENV "HOME"
             PATH_SUFFIXES onnxruntime/build/Linux/Release onnxruntime/build/Linux/Debug
             REQUIRED)

include(ExternalProject)

ExternalProject_Add(
    onnx_model
    DOWNLOAD_DIR ${PROJECT_SOURCE_DIR}/data
    URL https://github.com/onnx/models/raw/master/text/machine_comprehension/gpt-2/model/gpt2-lm-head-10.onnx
    DOWNLOAD_NO_EXTRACT FALSE
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

set(VOCAB_FILE ${PROJECT_SOURCE_DIR}/data/vocab.json)
set(MERGES_FILE ${PROJECT_SOURCE_DIR}/data/merges.txt)
set(ONNX_FILE ${PROJECT_SOURCE_DIR}/data/gpt2-lm-head-10.onnx)

configure_file(
    "${PROJECT_SOURCE_DIR}/src/config.hpp.in"
    "${PROJECT_BINARY_DIR}/src/config.hpp"
)

add_executable(gpt2-generate ${PROJECT_SOURCE_DIR}/src/main.cpp 
                             ${PROJECT_SOURCE_DIR}/src/simdjson.cpp)
target_include_directories(gpt2-generate PRIVATE 
                            ${ORT_INCLUDE_DIR}
                            ${PROJECT_BINARY_DIR}/src)
target_link_libraries(gpt2-generate PRIVATE ${ORT_LIBRARY})



# not that this would work on a windows machine..
if (MSVC)
    target_compile_options(gpt2-generate PRIVATE /W4 /WX)
else()
    target_compile_options(gpt2-generate PRIVATE -Wall -Wextra -Werror)
endif()