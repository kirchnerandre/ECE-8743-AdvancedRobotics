
#include <stdio.h>

#include "AlgorithPackageWrapping.h"


bool compare_vertices(VERTEX_T& VertexA, VERTEX_T& VertexB)
{
    if ((VertexA.X != VertexB.X) || (VertexA.Y != VertexB.Y))
    {
        fprintf(stderr, "%s:%d:%s: Vertices are different\n", __FILE__, __LINE__, __FUNCTION__);
        return false;
    }

    return true;
}


bool compare_vertices(VERTICES_T& VerticesA, VERTICES_T& VerticesB)
{
    if (VerticesA.size() != VerticesB.size())
    {
        fprintf(stderr, "%s:%d:%s: Vertices have different size\n", __FILE__, __LINE__, __FUNCTION__);
        return false;
    }

    for (size_t i = 0u; i < VerticesA.size(); i++)
    {
        if (!compare_vertices(VerticesA[i], VerticesB[i]))
        {
            fprintf(stderr, "%s:%d:%s: Vertices are different\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }
    }

    return true;
}


bool test_1()
{
    VERTICES_T vestices_input       = { { -10, -10 },
                                        {  10, -10 },
                                        {  10,  10 },
                                        { -10,  10 },
                                        { -20, -20 },
                                        {  20, -20 },
                                        {  20,  20 },
                                        { -20,  20 } };

    VERTICES_T vestices_expected    = { { -20, -20 },
                                        { -20,  20 },
                                        {  20,  20 },
                                        {  20, -20 } };

    VERTICES_T vertices_output;

    AlgorithPackageWrapping::compute(vertices_output, vestices_input);

    if (!compare_vertices(vertices_output, vestices_expected))
    {
        fprintf(stderr, "%s:%d:%s: Test failed\n", __FILE__, __LINE__, __FUNCTION__);
        return false;
    }

    return true;
}


int main()
{
    if (!test_1())
    {
        fprintf(stderr, "%s:%d:%s: Test failed\n", __FILE__, __LINE__, __FUNCTION__);
        return -1;
    }

    return 0;
}
