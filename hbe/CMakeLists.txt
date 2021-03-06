cmake_minimum_required(VERSION 3.5)
project(hbe)

set(CMAKE_CXX_STANDARD 14)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -fpic")


include_directories(utils)
include_directories(lib/config4cpp/include)
include_directories(lib/eigen-git-mirror)
add_subdirectory(alg)
add_subdirectory(data)
link_directories(lib/config4cpp/lib)

add_executable(hbe_exact main/ComputeExact.cpp)
#add_executable(hbe main/SketchBench.cpp)
add_executable(hbe_benchmark main/BatchBenchmark.cpp)
#add_executable(hbe main/Diagnosis.cpp)
#add_executable(hbe main/SyntheticDataGen.cpp)
add_executable(hbe_find main/FindAdaptiveEps.cpp)
add_executable(hbe main/RunAdaptive.cpp)
target_link_libraries(hbe alg data config4cpp)
target_link_libraries(hbe_exact alg data config4cpp)
target_link_libraries(hbe_benchmark alg data config4cpp)
target_link_libraries(hbe_find alg data config4cpp)

