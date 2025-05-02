
#ifndef _PRINTER_H_
#define _PRINTER_H_

#include <string>

#include "DataTypes.h"

struct BORDERS_T
{
    int32_t XMin;
    int32_t XMax;
    int32_t YMin;
    int32_t YMax;
};

bool print(
    std::string&    Filename,
    BORDERS_T&      Borders,
    VERTEX_T&       Begin,
    VERTEX_T&       End,
    EDGES_T&        Edges);

#endif // _PRINTER_H_
