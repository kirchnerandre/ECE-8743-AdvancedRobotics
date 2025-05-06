#ifndef _POLYGON_GPU_
#define _POLYGON_GPU_

#include "DataTypes.h"

bool test_colision(BOOLS_T& Statuses, VERTICES_T& Vertices, NUMBERS_T& PolygonsBegin, NUMBERS_T& PolygonsEnd, EDGE_T& Edge, size_t Vertices);

bool test_colision(BOOL_T* Statuses, NUMBER_T* PolygonsBegin, NUMBER_T* PolygonsEnd, VERTEX_T* Vertices, EDGE_T& Edge, size_t VerticesSize, size_t PolygonsSize);

#endif // _POLYGON_GPU_
