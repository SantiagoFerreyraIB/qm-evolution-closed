cmake_minimum_required(VERSION 3.10)
project(qm-evolution-closed LANGUAGES CXX)

# Require C++17 (you can bump to C++20 if needed)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Armadillo
find_package(Armadillo REQUIRED)

# Include headers
include_directories(${CMAKE_SOURCE_DIR}/include)

# Executable
add_executable(qm_example src/1qb_example.cpp)

# Link Armadillo
target_link_libraries(qm_example PRIVATE armadillo)
