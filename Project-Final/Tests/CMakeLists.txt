enable_testing()

add_executable              (TestsDijkstraGpu   TestsDijkstraGpu.cpp)
target_link_libraries       (TestsDijkstraGpu   PRIVATE ProjectFinal)
target_include_directories  (TestsDijkstraGpu   PRIVATE ../Headers)
add_test                    (TestsDijkstraGpu   TestsDijkstraGpu)

#add_executable              (TestsInterface     TestsInterface.cpp)
#target_link_libraries       (TestsInterface     PRIVATE ProjectFinal)
#target_include_directories  (TestsInterface     PRIVATE ../Headers)
#add_test                    (TestsInterface     TestsInterface)

#add_executable              (TestsPolygonCpu    TestsPolygonCpu.cpp)
#target_link_libraries       (TestsPolygonCpu    PRIVATE ProjectFinal)
#target_include_directories  (TestsPolygonCpu    PRIVATE ../Headers)
#add_test                    (TestsPolygonCpu    TestsPolygonCpu)

#add_executable              (TestsPolygonGpu    TestsPolygonGpu.cpp)
#target_link_libraries       (TestsPolygonGpu    PRIVATE ProjectFinal)
#target_include_directories  (TestsPolygonGpu    PRIVATE ../Headers)
#add_test                    (TestsPolygonGpu    TestsPolygonGpu)

#add_executable              (TestsPrinter       TestsPrinter.cpp)
#target_link_libraries       (TestsPrinter       PRIVATE ProjectFinal)
#target_include_directories  (TestsPrinter       PRIVATE ../Headers)
#add_test                    (TestsPrinter       TestsPrinter)
