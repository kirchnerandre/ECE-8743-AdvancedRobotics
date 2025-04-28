add_library(ProjectFinal STATIC
    PolygonCpu.cpp
    PolygonGpu.cu)

target_include_directories(ProjectFinal PRIVATE ../Headers)
