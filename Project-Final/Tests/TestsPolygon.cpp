
#include <stdio.h>

#include "PolygonCpu.h"


int main(int argc, char** argv)
{
    VERTICES_T  vertices;
    SIZES_T     sizes;

    if (!polygons_create(vertices, sizes, 0.0f, 5.0f, 0.0f, 5.0f, 10u, 5u))
    {
        return -1;
    }

    return 0;
}
