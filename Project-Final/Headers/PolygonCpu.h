#ifndef _POLYGON_CPU_
#define _POLYGON_CPU_

#include <vector>

struct VERTEX_T
{
    float X;
    float Y;
};

typedef std::vector<VERTEX_T>   POLYGOM_T;

bool polygon_create(POLYGOM_T& Polygon, VERTEX_T& VertexUpLeft, VERTEX_T& VertexLowerRight, size_t Vertices);

#endif // _POLYGON_CPU_
