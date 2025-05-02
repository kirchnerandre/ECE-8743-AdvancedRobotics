
#ifndef _DATA_TYPES_H_
#define _DATA_TYPES_H_

#include <stdint.h>
#include <vector>

struct VERTEX_T
{
    int32_t     X;
    int32_t     Y;
};

typedef std::vector<VERTEX_T>   VERTICES_T;
typedef VERTICES_T::iterator    IVERTEX_T;

struct EDGE_T
{
    VERTEX_T    VertexA;
    VERTEX_T    VertexB;
    bool        Status;
};

typedef std::vector<EDGE_T>     EDGES_T;
typedef EDGES_T::iterator       IEDGE_T;

#endif // _DATA_TYPES_H_
