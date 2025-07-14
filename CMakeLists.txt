cmake_minimum_required(VERSION 3.20)
project(GMRESSolver LANGUAGES CXX)

# Require C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Declare header-only interface library (no MPI linkage!)
add_library(gmres INTERFACE)
target_include_directories(gmres INTERFACE ${CMAKE_SOURCE_DIR}/include)

# Enable testing
enable_testing()

# Add each test directory with a main.cpp file
file(GLOB TEST_DIRS RELATIVE ${CMAKE_SOURCE_DIR}/test ${CMAKE_SOURCE_DIR}/test/*)

foreach(test_dir ${TEST_DIRS})
    if(IS_DIRECTORY ${CMAKE_SOURCE_DIR}/test/${test_dir})
        set(test_name ${test_dir})
        add_executable(${test_name} test/${test_dir}/main.cpp)
        target_link_libraries(${test_name} PRIVATE gmres)

        if(${test_name} STREQUAL "Laplacian1DParallel")
            target_link_libraries(${test_name} PRIVATE MPI::MPI_CXX)
        endif()

        add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
    endif()
endforeach()
