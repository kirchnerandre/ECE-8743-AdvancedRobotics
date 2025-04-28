#ifndef _POLYGON_CPU_
#define _POLYGON_CPU_

#include <vector>

struct VERTEX_T
{
    float X;
    float Y;
};

typedef std::vector<VERTEX_T> VERTICES_T;

typedef std::vector<uint32_t> SIZES_T;

void polygons_initialize();

bool polygons_create(
    VERTICES_T& Vertices,
    SIZES_T&    Sizes,
    float       XMin,
    float       XMax,
    float       YMin,
    float       YMax,
    uint32_t    PolygonsNumber,
    uint32_t    PolygonsSides);

#endif // _POLYGON_CPU_
