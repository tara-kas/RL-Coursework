cmake_minimum_required(VERSION 3.4...3.18)
project(mcts VERSION 0.0.1)


find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})


find_package(pybind11 QUIET)

if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found. Fetching from GitHub...")
    include(FetchContent)
    FetchContent_Declare(
      pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG v2.13.6  # Use a stable version
    )
    FetchContent_MakeAvailable(pybind11)
endif()

pybind11_add_module(mcts_cpp
  mcts_py.cpp
)

set_target_properties(mcts_cpp PROPERTIES PREFIX "")