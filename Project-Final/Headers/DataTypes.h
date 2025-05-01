
#ifndef _DATA_TYPES_H_
#define _DATA_TYPES_H_

#include <stdint.h>
#include <vector>

struct VERTEX_T
{
    int32_t X;
    int32_t Y;
};

typedef std::vector<VERTEX_T>   VERTICES_T;

#endif // _DATA_TYPES_H_
