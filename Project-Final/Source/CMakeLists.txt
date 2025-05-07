add_library(ProjectFinal STATIC
    Interface.cpp
    PolygonCpu.cpp
#   PolygonGpu.cu
    Printer.cpp
)

target_include_directories(ProjectFinal PRIVATE ../Headers)
