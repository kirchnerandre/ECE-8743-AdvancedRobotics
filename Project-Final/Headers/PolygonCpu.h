#ifndef _POLYGON_CPU_
#define _POLYGON_CPU_

#include "DataTypes.h"

bool test_colision(VERTICES_T& Vertices, NUMBER_T PolygonBegin, NUMBER_T PolygonEnd, EDGE_T& Edge);
bool test_colision(VERTEX_T*   Vertices, NUMBER_T PolygonBegin, NUMBER_T PolygonEnd, EDGE_T& Edge);

#endif // _POLYGON_CPU_
