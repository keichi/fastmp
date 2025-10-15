cmake_minimum_required(VERSION 3.15...4.1)
project(fastmp LANGUAGES CXX)

find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)

list(APPEND CMAKE_MODULE_PATH /opt/nec/ve/share/veoffload-veda/cmake)
list(APPEND CMAKE_PREFIX_PATH /opt/nec/ve3/lib)
find_package(VEDA)

enable_language(VEDA_CXX)
include_directories(${VEDA_INCLUDE_DIRS})
add_library(fastmp-device SHARED src/dot_product.vcpp src/stats.vcpp src/stomp.vcpp)

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_fastmp src/bindings.cpp)
target_link_libraries(_fastmp PRIVATE ${VEDA_LIBRARY})

install(TARGETS fastmp-device DESTINATION fastmp)
install(TARGETS _fastmp DESTINATION fastmp)
