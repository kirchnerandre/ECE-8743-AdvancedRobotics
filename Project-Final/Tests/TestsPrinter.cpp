
#include "Printer.h"


int main()
{
    std::string filename    = "./testfile1.ppm";
    BORDERS_T   borders     = { 0, 10, 0, 10 };
    VERTEX_T    begin       = { 1, 1 };
    VERTEX_T    end         = { 9, 9 };
    EDGES_T     edges;

    print(filename, borders, begin, end, edges);
}
