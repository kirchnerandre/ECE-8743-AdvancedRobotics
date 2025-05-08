
#include <cuda.h>

#include "DataTypes.h"


namespace
{
    constexpr int32_t max_threads   = 32;
    constexpr int32_t max_vertices  = 64;
    constexpr int32_t max_edges     = 1024;

    struct DATA_T
    {
        float   Cost;
        int32_t Source;
    };


    __device__ void print_data(DATA_T* Data)
    {
        if (threadIdx.x == 0)
        {
            for (int32_t i = 0; i < max_vertices * max_threads; i++)
            {
                if (Data[i].Cost > 0.0f)
                {
                    printf("%10d %5d %5d %10f %3d\n",
                        i,
                        i / max_vertices,
                        i % max_vertices,
                        Data[i].Cost,
                        Data[i].Source);
                }
            }

            printf("\n");
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


    __device__ void save_vertices(VERTEX_T* VerticesShared, VERTEX_T* VerticesGlobal, size_t VerticesSize)
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


    __device__ void load_vertices(VERTEX_T* VerticesShared, VERTEX_T* VerticesGlobal, size_t VerticesSize)
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


    __device__ void load_edges(EDGE_T* EdgesShared, EDGE_T* EdgesGlobal, size_t EdgesSize)
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


    __device__ void calculta_costs(DATA_T* Data, VERTEX_T* Vertices, EDGE_T* Edges, size_t VerticesSize, size_t EdgesSize)
    {
        for (int32_t i = 0; i < max_edges; i += max_threads)
        {
            if (threadIdx.x + i < EdgesSize)
            {
                if (Vertices[Edges[threadIdx.x + i].IndexA].Active)
                {
                    Data[Edges[threadIdx.x + i].IndexB + max_vertices * threadIdx.x].Cost   = Edges[threadIdx.x + i].Cost + Vertices[Edges[threadIdx.x + i].IndexA].Cost;
                    Data[Edges[threadIdx.x + i].IndexB + max_vertices * threadIdx.x].Source = Edges[threadIdx.x + i].IndexA;
#if 1
printf("%2d (%2d, %2d)\n", threadIdx.x, Edges[threadIdx.x + i].IndexA, Edges[threadIdx.x + i].IndexB);
#endif
                }
            }
        }

        __syncthreads();
    }


    __device__ void consolidate_data(DATA_T* Data, VERTEX_T* Vertices, EDGE_T* Edges, size_t VerticesSize, size_t EdgesSiz)
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
                    if (Data[threadIdx.x + i + max_vertices * j].Cost > 0.0f)
                    {
                        if (Vertices[threadIdx.x + i].Cost < 0.0f)
                        {
//if (threadIdx.x ==2) printf("A %2d %2d %5d\n", threadIdx.x, i, j);
                            Vertices[threadIdx.x + i].Cost    = Data[threadIdx.x + i + max_vertices * j].Cost;
                            Vertices[threadIdx.x + i].Source  = Data[threadIdx.x + i + max_vertices * j].Source;
                            Vertices[threadIdx.x + i].Active  = true;
//if (threadIdx.x ==2) printf("A %2d %2d %5d %10f, %2d, %d\n", threadIdx.x, i, j, Vertices[threadIdx.x + i].Cost, Vertices[threadIdx.x + i].Source, Vertices[threadIdx.x + i].Active);
                        }
                        else if (Vertices[threadIdx.x + i].Cost > Data[threadIdx.x + i + max_vertices * j].Cost)
                        {
//if (threadIdx.x ==2) printf("B %2d %2d\n", threadIdx.x, i);
                            Vertices[threadIdx.x + i].Cost    = Data[threadIdx.x + i + max_vertices * j].Cost;
                            Vertices[threadIdx.x + i].Source  = Data[threadIdx.x + i + max_vertices * j].Source;
                            Vertices[threadIdx.x + i].Active  = true;
                        }
                    }
                }
            }
        }

        __syncthreads();
    }


    __device__ void debug(VERTEX_T* Vertices, size_t VerticesSize)
    {
        if (threadIdx.x == 0)
        {
            for (int i = 0; i < VerticesSize; i++)
            {
                if (Vertices[i].Cost)
                {
                    printf("%4d %10f %3d %3d\n", i, Vertices[i].Cost, Vertices[i].Source, Vertices[i].Active);
                }
            }

            printf("\n");
        }

        __syncthreads();
    }


    __global__ void gpu_compute_path(VERTEX_T* Vertices, EDGE_T* Edges, size_t VerticesSize, size_t EdgesSize)
    {
        static     int32_t _counter = 2;

        __shared__  DATA_T      data    [max_vertices * max_threads];
        __shared__  VERTEX_T    vertices[max_vertices];
        __shared__  EDGE_T      edges   [max_edges];

        for (int32_t i = 0; i < max_vertices * max_threads; i += max_threads)
        {
            data[threadIdx.x + i] = { -1.0f, -1 };
        }

        load_vertices(vertices, Vertices, VerticesSize);

        load_edges(edges, Edges, EdgesSize);

        while (is_active(vertices, VerticesSize))
        {
            calculta_costs(data, vertices, edges, VerticesSize, EdgesSize);

print_data(data);

            consolidate_data(data, vertices, edges, VerticesSize, EdgesSize);

debug(vertices, VerticesSize);

//if (--_counter <= 0) break;
        }

        save_vertices(vertices, Vertices, VerticesSize);
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
