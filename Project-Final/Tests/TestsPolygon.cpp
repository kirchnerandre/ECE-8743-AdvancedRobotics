
#include <stdio.h>

#include "PolygonCpu.h"


int main(int argc, char** argv)
{
    POLYGOM_T   polygon;
    VERTEX_T    vertex_up_left;
    VERTEX_T    vertex_lower_right;
    size_t      vertices            = 10u;

    if (!polygon_create(polygon, vertex_up_left, vertex_lower_right, vertices))
    {
        return -1;
    }

    return 0;
}
