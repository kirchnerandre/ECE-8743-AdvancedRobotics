
#include <cuda.h>
#include <stdio.h>

#include "DijkstraGpu.h"


namespace
{
    constexpr int32_t max_threads   = 32;
    constexpr int32_t max_vertices  = 64;
    constexpr int32_t max_edges     = 1024;


    __device__ void save_vertices(
        VERTEX_T*   VerticesShared,
        VERTEX_T*   VerticesGlobal,
        size_t      VerticesSize)
    {
        for (int32_t i = 0; i < max_vertices; i += max_threads)
        {
            if (threadIdx.x + i < VerticesSize)
            {
                VerticesGlobal[threadIdx.x + i] = VerticesShared[threadIdx.x + i];
            }
        }

        __syncthreads();
    }


    __device__ void load_vertices(
        VERTEX_T*   VerticesShared,
        VERTEX_T*   VerticesGlobal,
        size_t      VerticesSize)
    {
        for (int32_t i = 0; i < max_vertices; i += max_threads)
        {
            if (threadIdx.x + i < VerticesSize)
            {
                VerticesShared[threadIdx.x + i] = VerticesGlobal[threadIdx.x + i];
            }
        }

        __syncthreads();
    }


    __device__ void load_edges(
        EDGE_T* EdgesShared,
        EDGE_T* EdgesGlobal,
        size_t  EdgesSize)
    {
        for (int32_t i = 0; i < max_edges; i += max_threads)
        {
            if (threadIdx.x < EdgesSize)
            {
                EdgesShared[threadIdx.x + i] = EdgesGlobal[threadIdx.x + i];
            }
        }

        __syncthreads();
    }


    __device__ bool is_active(VERTEX_T* Vertices, size_t VerticesSize)
    {
        __shared__ bool active[max_vertices];

        for (int32_t i = 0; i < max_vertices; i += max_threads)
        {
            active[threadIdx.x + i] = Vertices[threadIdx.x + i].Active;
        }

        __syncthreads();

        for (int32_t i = max_vertices / 2; i >= 1; i /= 2)
        {
            if (threadIdx.x < i)
            {
                if ((active[threadIdx.x]) || (active[threadIdx.x + i]))
                {
                    active[threadIdx.x] = true;
                }
            }
        }

        __syncthreads();

        return active[0];
    }


    __device__ void calculate_costs(
        VERTEX_T*   VerticesTmp,
        VERTEX_T*   Vertices,
        EDGE_T*     Edges,
        size_t      VerticesSize,
        size_t      EdgesSize)
    {
        for (int32_t i = 0; i < max_edges; i += max_threads)
        {
            if (threadIdx.x + i < EdgesSize)
            {
                if (Vertices[Edges[threadIdx.x + i].IndexA].Active)
                {
                    VerticesTmp[Edges[threadIdx.x + i].IndexB + max_vertices * threadIdx.x] = {
                        Edges[threadIdx.x + i].IndexA,
                        true,
                        Edges[threadIdx.x + i].Cost + Vertices[Edges[threadIdx.x + i].IndexA].Cost
                    };
                }
                else if (Vertices[Edges[threadIdx.x + i].IndexB].Active)
                {
                    VerticesTmp[Edges[threadIdx.x + i].IndexA + max_vertices * threadIdx.x] = {
                        Edges[threadIdx.x + i].IndexB,
                        true,
                        Edges[threadIdx.x + i].Cost + Vertices[Edges[threadIdx.x + i].IndexB].Cost
                    };
                }
            }
        }

        __syncthreads();
    }


    __device__ void consolidate_costs(
        VERTEX_T*   VerticesTmp,
        VERTEX_T*   Vertices,
        EDGE_T*     Edges,
        size_t      VerticesSize,
        size_t      EdgesSiz)
    {
        for (int32_t i = 0; i < max_vertices; i += max_threads)
        {
            if (threadIdx.x + i < VerticesSize)
            {
                Vertices[threadIdx.x + i].Active = false;
            }

            for (int32_t j = 0; j < max_threads; j++)
            {
                if (threadIdx.x + i < VerticesSize)
                {
                    if (VerticesTmp[threadIdx.x + i + max_vertices * j].Cost > 0.0f)
                    {
                        if (Vertices[threadIdx.x + i].Cost < 0.0f)
                        {
                            Vertices[threadIdx.x + i] = {
                                VerticesTmp[threadIdx.x + i + max_vertices * j].Previous,
                                true,
                                VerticesTmp[threadIdx.x + i + max_vertices * j].Cost
                            };
                        }
                        else if (Vertices[threadIdx.x + i].Cost >
                            VerticesTmp[threadIdx.x + i + max_vertices * j].Cost)
                        {
                            Vertices[threadIdx.x + i] = {
                                VerticesTmp[threadIdx.x + i + max_vertices * j].Previous,
                                true,
                                VerticesTmp[threadIdx.x + i + max_vertices * j].Cost
                            };
                        }
                    }
                }
            }
        }

        __syncthreads();
    }


    __global__ void gpu_compute_path(
        VERTEX_T*   Vertices,
        EDGE_T*     Edges,
        size_t      VerticesSize,
        size_t      EdgesSize)
    {
        __shared__  VERTEX_T    vertices_tmp[max_vertices * max_threads];
        __shared__  VERTEX_T    vertices    [max_vertices];
        __shared__  EDGE_T      edges       [max_edges];

        for (int32_t i = 0; i < max_vertices * max_threads; i += max_threads)
        {
            vertices_tmp[threadIdx.x + i] = { -1, false, -1.0f };
        }

        load_vertices(vertices, Vertices, VerticesSize);

        load_edges(edges, Edges, EdgesSize);

        while (is_active(vertices, VerticesSize))
        {
            calculate_costs(vertices_tmp, vertices, edges, VerticesSize, EdgesSize);

            consolidate_costs(vertices_tmp, vertices, edges, VerticesSize, EdgesSize);
        }

        save_vertices(vertices, Vertices, VerticesSize);
    }
}


bool compute_path(
    VERTEX_T*   Vertices,
    EDGE_T*     Edges,
    size_t      VerticesSize,
    size_t      EdgesSize)
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

    if (cudaMemcpy(Vertices, gpu_vertices, VerticesSize * sizeof(VERTEX_T), cudaMemcpyDeviceToHost))
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
