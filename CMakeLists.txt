cmake_minimum_required(VERSION 3.15...4.1)
project(fastmp LANGUAGES CXX)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -ffast-math")

if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -qopt-report=max -qopt-report-phase=vec")
endif()

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

add_library(fastmp src/dot_product.cpp src/stats.cpp src/stomp.cpp)

if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    # zmm-usage=high causes pocketfft to crash
    set_source_files_properties(src/stomp.cpp PROPERTIES COMPILE_FLAGS "-qopt-zmm-usage=high")
endif()

set_target_properties(fastmp PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(bench src/bench.cpp)
target_link_libraries(bench PRIVATE fastmp)

nanobind_add_module(_fastmp src/fastmp.cpp)
target_link_libraries(_fastmp PRIVATE fastmp)

install(TARGETS _fastmp DESTINATION fastmp)
