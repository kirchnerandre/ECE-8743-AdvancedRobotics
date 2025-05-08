add_library(ProjectFinal STATIC
    DijkstraGpu.cu
)

target_include_directories(ProjectFinal PRIVATE ../Headers)
