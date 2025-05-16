
#ifndef _DIJKSTRA_GPU_
#define _DIJKSTRA_GPU_

#include <stdint.h>

struct VERTEX_T
{
    int32_t     Previous;
    bool        Active;
    float       Cost;
};

struct EDGE_T
{
    int32_t     IndexA;
    int32_t     IndexB;
    float       Cost;
};

bool compute_path(
    VERTEX_T*   Vertices,
    EDGE_T*     Edges,
    size_t      VerticesSize,
    size_t      EdgesSize);

#endif // _DIJKSTRA_GPU_
