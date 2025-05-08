enable_testing()

add_executable              (TestsDijkstraGpu   TestsDijkstraGpu.cpp)
target_link_libraries       (TestsDijkstraGpu   PRIVATE ProjectFinal)
target_include_directories  (TestsDijkstraGpu   PRIVATE ../Headers)
add_test                    (TestsDijkstraGpu   TestsDijkstraGpu)

