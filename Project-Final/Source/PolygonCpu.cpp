
#include <cstdlib>

#include "PolygonCpu.h"


namespace
{
    bool polygons_create()
    {
        return true;
    }
}


void polygon_initialize()
{
    std::srand(0u);
}


bool polygons_create(
    VERTICES_T& Vertices,
    SIZES_T&    Sizes,
    float       XMin,
    float       XMax,
    float       YMin,
    float       YMax,
    uint32_t    PolygonsNumber,
    uint32_t    PolygonsSides)
{
    Sizes.resize(PolygonsNumber, 0u);

    for (uint32_t i = 0u; i < PolygonsNumber; i++)
    {
        Sizes[i] = 3u + std::rand() % (PolygonsSides - 3u);
    }

    uint32_t vertices_total = 0u;

    for (uint32_t i = 0u; i < PolygonsNumber; i++)
    {
        vertices_total += Sizes[i];
    }

    Vertices.resize(vertices_total, { 0.0f, 0.0f });

    return true;
}
