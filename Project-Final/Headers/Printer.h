
#ifndef _PRINTER_H_
#define _PRINTER_H_

#include <string>

#include "DataTypes.h"

bool print(
    std::string&    Filename,
    VERTEX_T&       Begin,
    VERTEX_T&       End,
    EDGES_T&        Edges);

#endif // _PRINTER_H_
