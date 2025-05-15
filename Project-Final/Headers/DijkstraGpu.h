
#ifndef _DIJKSTRA_GPU_
#define _DIJKSTRA_GPU_

#include "DataTypes.h"

bool compute_path(
    VERTEX_T*   Vertices,
    EDGE_T*     Edges,
    size_t      VerticesSize,
    size_t      EdgesSize);

#endif // _DIJKSTRA_GPU_
