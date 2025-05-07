
#ifndef _DATA_TYPES_H_
#define _DATA_TYPES_H_

#include <stdint.h>
#include <vector>

struct VERTEX_T
{
    float       X;
    float       Y;
};

typedef std::vector<VERTEX_T>   VERTICES_T;
typedef VERTICES_T::iterator    IVERTEX_T;

struct EDGE_T
{
    int32_t     IndexA;
    int32_t     IndexB;
    bool        Status;
};

typedef std::vector<EDGE_T>     EDGES_T;
typedef EDGES_T::iterator       IEDGE_T;

typedef int32_t                 NUMBER_T;
typedef std::vector<NUMBER_T>   NUMBERS_T;
typedef NUMBERS_T::iterator     INUMBER_T;

typedef bool                    BOOL_T;
typedef std::vector<BOOL_T>     BOOLS_T;
typedef BOOLS_T::iterator       IBOOL_T;

#endif // _DATA_TYPES_H_
