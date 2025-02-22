
enable_testing ()

set (TEST_NAME "test-001")
add_executable(
    ${TEST_NAME}
    TestAlgorithPackageWrapping.cpp)
include_directories(
    ${PROJECT_BASEDIR}/Headers)
target_link_libraries(
    ${TEST_NAME}
    ${PROJECT_NAME})
add_test(
    NAME                ${TEST_NAME}
    COMMAND             ${TEST_NAME}
    WORKING_DIRECTORY   ${PROJECT_BASEDIR})
set_property(
    TARGET              ${TEST_NAME}
    PROPERTY            CXX_STANDARD 17)
