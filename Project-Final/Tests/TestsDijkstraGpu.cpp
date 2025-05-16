
#include <stdio.h>

#include "DijkstraGpu.h"


namespace
{
    bool compare_paths(VERTEX_T* VerticesA, VERTEX_T* VerticesB, size_t VerticesSize)
    {
        for (size_t i = 0u; i < VerticesSize; i++)
        {
            if ((VerticesA[i].Previous  != VerticesB[i].Previous)
             || (VerticesA[i].Active    != VerticesB[i].Active)
             || (VerticesA[i].Cost      != VerticesB[i].Cost))
            {
                fprintf(stderr, "%s:%d:%s: Invalid vertex\n", __FILE__, __LINE__, __FUNCTION__);
                return false;
            }
        }

        return true;
    }


    void print_path(VERTEX_T* Vertices, size_t VerticesSize, int32_t VertexEnd)
    {
        int32_t source = VertexEnd;

        printf(" %10f:", Vertices[VertexEnd].Cost);

        while (1)
        {
            if (source >= static_cast<int32_t>(VerticesSize))
            {
                break;
            }

            if (source < 0)
            {
                break;
            }
            else
            {
                printf(" %2d", source);
            }

            source = Vertices[source].Previous;
        }

        printf("\n");
    }


    bool test(
        int32_t     VertexEnd,
        VERTEX_T*   Expected,
        VERTEX_T*   Vertices,
        EDGE_T*     Edges,
        size_t      VerticesSize,
        size_t      EdgesSize)
    {
        if (!compute_path(Vertices, Edges, VerticesSize, EdgesSize))
        {
            fprintf(stderr, "%s:%d:%s: Failed to compute path\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        print_path(Vertices, VerticesSize, VertexEnd);

        if (!compare_paths(Vertices, Expected, VerticesSize))
        {
            fprintf(stderr, "%s:%d:%s: Invalid path\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        return true;
    }
}


int main()
{
    VERTEX_T    expected_1[]    = {
        { -1, false,   0.0f },
        {  0, false, 100.0f },
        {  0, false, 100.0f },
        {  0, false, 141.0f }
    };

    VERTEX_T    vertices_1[]    = {
        { -1, true,   0.0f },
        { -1, false, -1.0f },
        { -1, false, -1.0f },
        { -1, false, -1.0f }
    };

    EDGE_T      edges_1[]       = {
        { 1,  0,  100.0f },
        { 2,  1,  141.0f },
        { 3,  2,  100.0f },
        { 3,  0,  141.0f },
        { 2,  0,  100.0f },
        { 3,  1,  100.0f },
    };

    VERTEX_T    expected_2[]    = {
        { -1, false,   0.0f },
        {  0, false, 100.0f },
        {  1, false, 200.0f },
        {  0, false, 100.0f },
        {  0, false, 141.0f },
        {  4, false, 241.0f },
        {  3, false, 200.0f },
        {  4, false, 241.0f },
        {  4, false, 282.0f }
    };

    VERTEX_T    vertices_2[]    = {
        { -1, true,   0.0f },
        { -1, false, -1.0f },
        { -1, false, -1.0f },
        { -1, false, -1.0f },
        { -1, false, -1.0f },
        { -1, false, -1.0f },
        { -1, false, -1.0f },
        { -1, false, -1.0f },
        { -1, false, -1.0f }
    };

    EDGE_T      edges_2[]       = {
        { 0,  1,  100.0f },
        { 1,  2,  100.0f },
        { 3,  4,  100.0f },
        { 4,  5,  100.0f },
        { 6,  7,  100.0f },
        { 7,  8,  100.0f },
        { 0,  3,  100.0f },
        { 1,  4,  100.0f },
        { 2,  5,  100.0f },
        { 3,  6,  100.0f },
        { 4,  7,  100.0f },
        { 5,  8,  100.0f },
        { 0,  4,  141.0f },
        { 1,  5,  141.0f },
        { 3,  7,  141.0f },
        { 4,  8,  141.0f },
        { 3,  1,  141.0f },
        { 4,  2,  141.0f },
        { 6,  4,  141.0f },
        { 7,  5,  141.0f }
    };

    VERTEX_T    expected_3[]    = {
        { -1, false,   0.0f },
        {  0, false, 100.0f },
        {  1, false, 200.0f },
        {  2, false, 300.0f },
        {  0, false, 100.0f },
        {  0, false, 141.0f },
        {  5, false, 241.0f },
        {  6, false, 341.0f },
        {  4, false, 200.0f },
        {  5, false, 241.0f },
        {  5, false, 282.0f },
        { 10, false, 382.0f },
        {  8, false, 300.0f },
        {  9, false, 341.0f },
        { 10, false, 382.0f },
        { 10, false, 423.0f }
    };

    VERTEX_T    vertices_3[]    = {
        { -1, true,   0.0f },
        { -1, false, -1.0f },
        { -1, false, -1.0f },
        { -1, false, -1.0f },
        { -1, false, -1.0f },
        { -1, false, -1.0f },
        { -1, false, -1.0f },
        { -1, false, -1.0f },
        { -1, false, -1.0f },
        { -1, false, -1.0f },
        { -1, false, -1.0f },
        { -1, false, -1.0f },
        { -1, false, -1.0f },
        { -1, false, -1.0f },
        { -1, false, -1.0f },
        { -1, false, -1.0f }
    };

    EDGE_T      edges_3[]       = {
        { 0,  1,  100.0f },
        { 1,  2,  100.0f },
        { 2,  3,  100.0f },
        { 4,  5,  100.0f },
        { 5,  6,  100.0f },
        { 6,  7,  100.0f },
        { 8,  9,  100.0f },
        { 9,  10, 100.0f },
        { 10, 11, 100.0f },
        { 12, 13, 100.0f },
        { 13, 14, 100.0f },
        { 14, 15, 100.0f },
        { 0,  4,  100.0f },
        { 4,  8,  100.0f },
        { 8,  12, 100.0f },
        { 1,  5,  100.0f },
        { 5,  9,  100.0f },
        { 9,  13, 100.0f },
        { 2,  6,  100.0f },
        { 6,  10, 100.0f },
        { 10, 14, 100.0f },
        { 3,  7,  100.0f },
        { 7,  11, 100.0f },
        { 11, 15, 100.0f },
        { 0,  5,  141.0f },
        { 4,  9,  141.0f },
        { 8,  13, 141.0f },
        { 1,  6,  141.0f },
        { 5,  10, 141.0f },
        { 9,  14, 141.0f },
        { 2,  7,  141.0f },
        { 6,  11, 141.0f },
        { 10, 15, 141.0f },
        { 1,  4,  141.0f },
        { 5,  8,  141.0f },
        { 9,  12, 141.0f },
        { 2,  5,  141.0f },
        { 6,  9,  141.0f },
        { 10, 13, 141.0f },
        { 3,  6,  141.0f },
        { 7,  10, 141.0f },
        { 11, 14, 141.0f },
    };

    int32_t     Vertex_end_1    = 3;
    int32_t     Vertex_end_2    = 7;
    int32_t     Vertex_end_3    = 14;

    if (!test(
        Vertex_end_1,
        expected_1,
        vertices_1,
        edges_1,
        sizeof(vertices_1)  / sizeof(VERTEX_T),
        sizeof(edges_1)     / sizeof(EDGE_T)))
    {
        fprintf(stderr, "%s:%d:%s: Test failed\n", __FILE__, __LINE__, __FUNCTION__);
        return -1;
    }

    if (!test(
        Vertex_end_2,
        expected_2,
        vertices_2,
        edges_2,
        sizeof(vertices_2)  / sizeof(VERTEX_T),
        sizeof(edges_2)     / sizeof(EDGE_T)))
    {
        fprintf(stderr, "%s:%d:%s: Test failed\n", __FILE__, __LINE__, __FUNCTION__);
        return -1;
    }

    if (!test(
        Vertex_end_3,
        expected_3,
        vertices_3,
        edges_3,
        sizeof(vertices_3)  / sizeof(VERTEX_T),
        sizeof(edges_3)     / sizeof(EDGE_T)))
    {
        fprintf(stderr, "%s:%d:%s: Test failed\n", __FILE__, __LINE__, __FUNCTION__);
        return -1;
    }

    return 0;
}
