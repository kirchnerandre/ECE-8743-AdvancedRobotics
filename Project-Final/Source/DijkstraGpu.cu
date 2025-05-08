
#include <cuda.h>

#include "DataTypes.h"


namespace
{
    constexpr int32_t max_threads   = 32;
    constexpr int32_t max_vertices  = 64;
    constexpr int32_t max_edges     = 1024;

    struct PATH_T
    {
        float   Cost;
        int32_t Index;
    };

    __global__ void gpu_compute_path(VERTEX_T* Vertices, EDGE_T* Edges, size_t VerticesSize, size_t EdgesSize)
    {
        __shared__  PATH_T      paths_all   [max_vertices * max_threads];
        __shared__  VERTEX_T    vertices    [max_vertices];
        __shared__  EDGE_T      edges       [max_edges];

        int32_t                 x           = threadIdx.x;

        for (int32_t i = 0; i < max_vertices; i += max_threads)
        {
            if (x < VerticesSize)
            {
                vertices[x + i] = Vertices[x + i];
            }
        }

        __syncthreads();

        for (int32_t i = 0; i < max_edges; i += max_threads)
        {
            if (x < EdgesSize)
            {
                edges[x + i] = Edges[x + i];
            }
        }

        __syncthreads();

        for (int32_t i = 0; i < max_edges; i += max_threads)
        {
            if (x + i < EdgesSize)
            {
                if (vertices[Edges[x + i].IndexA].Active)
                {
printf("B %d %d %d\n", x, i, Edges[x].IndexB);
                    paths_all[Edges[x].IndexB + max_vertices * x] = { vertices[Edges[x + i].IndexA].Cost + edges[x + i].Cost, Edges[x + i].IndexA };
                }
            }
        }

        __syncthreads();

if (x == 0)
for (int i = 0; i < max_vertices * max_threads; i++)
{
    if (paths_all[i].Cost)
    {
        printf("%4d %f %d\n", i, paths_all[i].Cost, paths_all[i].Index);
    }
}
__syncthreads();

        __syncthreads();

        for (int32_t i = 0; i < max_vertices; i += max_threads)
        {
            if (x + i < VerticesSize)
            {
                vertices[x + i].Active = false;
            }

            for (int32_t j = 0; j < max_threads; j++)
            {
                if (x + i < VerticesSize)
                {
                    if (paths_all[x + i + max_vertices * j].Cost > 0.0f)
                    {
                        if (vertices[x + i].Cost == 0.0f)
                        {
printf("A %d %10f %10f\n", x + i, vertices[x + i].Cost, paths_all[x + i + max_vertices * j].Cost);
                            vertices[x + i].Cost    = paths_all[x + i + max_vertices * j].Cost;
                            vertices[x + i].Index   = paths_all[x + i + max_vertices * j].Index;
                            vertices[x + i].Active  = true;
                        }
                        else if (vertices[x + i].Cost > paths_all[x + i + max_vertices * j].Cost)
                        {
printf("B %d %10f %10f\n", x + i, vertices[x + i].Cost, paths_all[x + i + max_vertices * j].Cost);
                            vertices[x + i].Cost    = paths_all[x + i + max_vertices * j].Cost;
                            vertices[x + i].Index   = paths_all[x + i + max_vertices * j].Index;
                            vertices[x + i].Active  = true;
                        }
                    }
                }
            }
        }

        __syncthreads();

if (x == 0)
for (int i = 0; i < max_vertices; i++)
{
    if (vertices[i].Cost)
    {
        printf("Z %4d %f %d %d\n", i, vertices[i].Cost, vertices[i].Index, vertices[i].Active);
    }
}
__syncthreads();
    }
}


bool compute_path(VERTEX_T* Vertices, EDGE_T* Edges, size_t VerticesSize, size_t EdgesSize)
{
    bool        ret_val         = true;
    VERTEX_T*   gpu_vertices    = nullptr;
    EDGE_T*     gpu_edges       = nullptr;

    dim3 blocks     = { 1,              1 };
    dim3 threads    = { max_threads,    1 };

    if (cudaMalloc(&gpu_vertices, VerticesSize * sizeof(VERTEX_T)))
    {
        fprintf(stderr, "%s:%d:%s: Failed to allocate memory\n", __FILE__, __LINE__, __FUNCTION__);
        ret_val = false;
        goto terminate;
    }

    if (cudaMalloc(&gpu_edges, EdgesSize * sizeof(EDGE_T)))
    {
        fprintf(stderr, "%s:%d:%s: Failed to allocate memory\n", __FILE__, __LINE__, __FUNCTION__);
        ret_val = false;
        goto terminate;
    }

    if (cudaMemcpy(gpu_vertices, Vertices, VerticesSize * sizeof(VERTEX_T), cudaMemcpyHostToDevice))
    {
        fprintf(stderr, "%s:%d:%s: Failed to copy memory\n", __FILE__, __LINE__, __FUNCTION__);
        ret_val = false;
        goto terminate;
    }

    if (cudaMemcpy(gpu_edges, Edges, EdgesSize * sizeof(EDGE_T), cudaMemcpyHostToDevice))
    {
        fprintf(stderr, "%s:%d:%s: Failed to copy memory\n", __FILE__, __LINE__, __FUNCTION__);
        ret_val = false;
        goto terminate;
    }

    gpu_compute_path<<<blocks, threads>>>(gpu_vertices, gpu_edges, VerticesSize, EdgesSize);

    if (cudaMemcpy(Edges, gpu_edges, EdgesSize * sizeof(EDGE_T), cudaMemcpyDeviceToHost))
    {
        fprintf(stderr, "%s:%d:%s: Failed to copy memory\n", __FILE__, __LINE__, __FUNCTION__);
        ret_val = false;
        goto terminate;
    }

terminate:
    if (gpu_vertices)
    {
        cudaFree(gpu_vertices);
    }

    if (gpu_edges)
    {
        cudaFree(gpu_edges);
    }

    return ret_val;
}
