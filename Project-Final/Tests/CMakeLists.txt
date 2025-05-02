enable_testing()

add_executable              (TestsInterface TestsInterface.cpp)
target_link_libraries       (TestsInterface PRIVATE ProjectFinal)
target_include_directories  (TestsInterface PRIVATE ../Headers)
add_test                    (TestsInterface TestsInterface)

add_executable              (TestsPolygon TestsPolygon.cpp)
target_link_libraries       (TestsPolygon PRIVATE ProjectFinal)
target_include_directories  (TestsPolygon PRIVATE ../Headers)
add_test                    (TestsPolygon TestsPolygon)

add_executable              (TestsPrinter TestsPrinter.cpp)
target_link_libraries       (TestsPrinter PRIVATE ProjectFinal)
target_include_directories  (TestsPrinter PRIVATE ../Headers)
add_test                    (TestsPrinter TestsPrinter)

add_executable              (TestsUtils TestsUtils.cpp)
target_link_libraries       (TestsUtils PRIVATE ProjectFinal)
target_include_directories  (TestsUtils PRIVATE ../Headers)
add_test                    (TestsUtils TestsUtils)
