enable_testing()

add_executable              (TestsPolygon TestsPolygon.cpp)
target_link_libraries       (TestsPolygon PRIVATE ProjectFinal)
target_include_directories  (TestsPolygon PRIVATE ../Headers)
add_test                    (TestsPolygon TestsPolygon)
