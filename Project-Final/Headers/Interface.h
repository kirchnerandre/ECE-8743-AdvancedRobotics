
#ifndef _INTERFACE_H_
#define _INTERFACE_H_

#include <string>

#include "DataTypes.h"

bool read_map(VERTICES_T& Vertices, size_t& Polygons, std::string Filename);

bool read_map(VERTICES_T& Vertices, EDGES_T& Edges, NUMBERS_T& PolygonsBegin, NUMBERS_T& PolygonsEnd, std::string Filename);

#endif // _INTERFACE_H_
