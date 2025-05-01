
#include "Utils.h"


namespace
{
    __global__ void compute_colision(bool* Results, void* Data)
    {
        uint32_t x          = threadIdx.x;
        uint32_t polygons   = *static_cast<uint32_t*>(Data);
        uint32_t offset     = sizeof(uint32_t);

        if (x < polygons)
        {
            offset += *static_cast<uint32_t*>(Data[offset + x]);

            uint32_t vertices = *static_cast<uint32_t*>(Data[offset]);

            offset += sizeof(uint32_t);

            for (uint32_t i = 1u; i < vertices; i++)
            {

            }
        }
    }
}
