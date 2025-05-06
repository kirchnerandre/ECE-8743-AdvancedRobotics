enable_testing()

add_executable              (TestsInterface     TestsInterface.cpp)
target_link_libraries       (TestsInterface     PRIVATE ProjectFinal)
target_include_directories  (TestsInterface     PRIVATE ../Headers)
add_test                    (TestsInterface     TestsInterface)

add_executable              (TestsPolygonCpu    TestsPolygonCpu.cpp)
target_link_libraries       (TestsPolygonCpu    PRIVATE ProjectFinal)
target_include_directories  (TestsPolygonCpu    PRIVATE ../Headers)
add_test                    (TestsPolygonCpu    TestsPolygonCpu)

add_executable              (TestsPrinter       TestsPrinter.cpp)
target_link_libraries       (TestsPrinter       PRIVATE ProjectFinal)
target_include_directories  (TestsPrinter       PRIVATE ../Headers)
add_test                    (TestsPrinter       TestsPrinter)

add_executable              (TestsUtils         TestsUtils.cpp)
target_link_libraries       (TestsUtils         PRIVATE ProjectFinal)
target_include_directories  (TestsUtils         PRIVATE ../Headers)
add_test                    (TestsUtils         TestsUtils)
