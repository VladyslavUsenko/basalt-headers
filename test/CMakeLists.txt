cmake_minimum_required(VERSION 3.2)

if(NOT TARGET gtest_main)
    add_subdirectory(googletest gtest EXCLUDE_FROM_ALL)
    include_directories(googletest)
endif(NOT TARGET gtest_main)
set(GTEST_MAIN_LIBRARY gtest_main)
set(GTEST_LIBRARY gtest)


include_directories(include)

add_executable(test_spline src/test_spline.cpp)
target_link_libraries(test_spline ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY})

add_executable(test_spline_se3 src/test_spline_se3.cpp)
target_link_libraries(test_spline_se3 ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY})

add_executable(test_camera src/test_camera.cpp)
target_link_libraries(test_camera ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY})

add_executable(test_sophus src/test_sophus.cpp)
target_link_libraries(test_sophus ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY})

add_executable(test_preintegration src/test_preintegration.cpp)
target_link_libraries(test_preintegration ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} ${TBB_LIBRARIES})

# benchmarks
IF(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "ENABLE tests")
    set(BENCHMARK_ENABLE_EXCEPTIONS OFF CACHE BOOL "ENABLE exceptions")
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "ENABLE gtests")
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "ENABLE install")
    add_subdirectory(benchmark)
    include_directories(benchmark/include)

    add_executable(benchmark_camera src/benchmark_camera.cpp)
    target_link_libraries(benchmark_camera benchmark)
ENDIF()

enable_testing()

add_test(test_spline test_spline COMMAND Test)
add_test(test_camera test_camera COMMAND Test)
add_test(test_sophus test_sophus COMMAND Test)
add_test(test_preintegration test_preintegration COMMAND Test)
