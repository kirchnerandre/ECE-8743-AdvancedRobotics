
include_directories(
    ${PROJECT_BASEDIR}/Headers
)

add_library(${PROJECT_NAME} STATIC
    AlgorithPackageWrapping.cpp)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
