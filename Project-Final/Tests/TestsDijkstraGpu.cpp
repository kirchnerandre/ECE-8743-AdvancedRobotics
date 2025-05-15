
#include <stdio.h>

#include "DijkstraGpu.h"


namespace
{
    bool compare_paths(VERTEX_T* VerticesA, VERTEX_T* VerticesB)
    {
        return true;
    }


    void print_path(VERTEX_T* Vertices, int32_t VertexEnd)
    {
        int32_t source = VertexEnd;

        printf(" %10f", Vertices[VertexEnd].Cost);

        while (1)
        {
            if (source < 0)
            {
                break;
            }
            else
            {
                printf(" %2d", source);
            }

            source = Vertices[source].Source;
        }

        printf("\n");
    }


    bool test_1()
    {
        VERTEX_T    vertices[]  = {
            { -1, true,   0.0f },
            { -1, false, -1.0f },
            { -1, false, -1.0f },
            { -1, false, -1.0f } };

        EDGE_T      edges[]     = {
            { 0,  1,  100.0f },
            { 1,  2,  141.0f },
            { 2,  3,  100.0f },
            { 3,  0,  141.0f },
            { 0,  2,  100.0f },
            { 1,  3,  100.0f },

            { 1,  0,  100.0f },
            { 2,  1,  141.0f },
            { 3,  2,  100.0f },
            { 0,  3,  141.0f },
            { 2,  0,  100.0f },
            { 3,  1,  100.0f },
        };

        if (!compute_path(
            vertices,
            edges,
            sizeof(vertices) / sizeof(VERTEX_T),
            sizeof(edges) / sizeof(EDGE_T)))
        {
            fprintf(stderr, "%s:%d:%s: Failed to compute path\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        print_path(vertices, 3);

        return true;
    }


    bool test_2()
    {
        VERTEX_T    vertices[]  = {
            { -1, true,   0.0f }, { -1, false, -1.0f }, { -1, false, -1.0f },
            { -1, false, -1.0f }, { -1, false, -1.0f }, { -1, false, -1.0f },
            { -1, false, -1.0f }, { -1, false, -1.0f }, { -1, false, -1.0f } };

        EDGE_T      edges[]     = {
            { 0,  1,  100.0f }, { 1,  2,  100.0f },
            { 3,  4,  100.0f }, { 4,  5,  100.0f },
            { 6,  7,  100.0f }, { 7,  8,  100.0f },

            { 0,  3,  100.0f }, { 1,  4,  100.0f }, { 2,  5,  100.0f },
            { 3,  6,  100.0f }, { 4,  7,  100.0f }, { 5,  8,  100.0f },

            { 0,  4,  141.0f }, { 1,  5,  141.0f },
            { 3,  7,  141.0f }, { 4,  8,  141.0f },

            { 3,  1,  141.0f }, { 4,  2,  141.0f },
            { 6,  4,  141.0f }, { 7,  5,  141.0f },

            { 1,  0,  100.0f }, { 2,  1,  100.0f },
            { 4,  3,  100.0f }, { 5,  4,  100.0f },
            { 7,  6,  100.0f }, { 8,  7,  100.0f },

            { 3,  0,  100.0f }, { 4,  1,  100.0f }, { 5,  2,  100.0f },
            { 6,  3,  100.0f }, { 7,  4,  100.0f }, { 8,  5,  100.0f },

            { 4,  0,  141.0f }, { 5,  1,  141.0f },
            { 7,  3,  141.0f }, { 8,  4,  141.0f },

            { 1,  3,  141.0f }, { 2,  4,  141.0f },
            { 4,  6,  141.0f }, { 5,  7,  141.0f },
        };

        if (!compute_path(vertices, edges, sizeof(vertices) / sizeof(VERTEX_T), sizeof(edges) / sizeof(EDGE_T)))
        {
            fprintf(stderr, "%s:%d:%s: Failed to compute path\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        print_path(vertices, 7);

        return true;
    }


    bool test_3()
    {
        VERTEX_T    vertices[]  = {
            { -1, true,   0.0f }, { -1, false, -1.0f },
            { -1, false, -1.0f }, { -1, false, -1.0f },
            { -1, false, -1.0f }, { -1, false, -1.0f },
            { -1, false, -1.0f }, { -1, false, -1.0f },
            { -1, false, -1.0f }, { -1, false, -1.0f },
            { -1, false, -1.0f }, { -1, false, -1.0f },
            { -1, false, -1.0f }, { -1, false, -1.0f },
            { -1, false, -1.0f }, { -1, false, -1.0f } };

        EDGE_T      edges[]     = {
            { 0,  1,  100.0f }, { 1,  2,  100.0f }, { 2,  3,  100.0f },
            { 4,  5,  100.0f }, { 5,  6,  100.0f }, { 6,  7,  100.0f },
            { 8,  9,  100.0f }, { 9,  10, 100.0f }, { 10, 11, 100.0f },
            { 12, 13, 100.0f }, { 13, 14, 100.0f }, { 14, 15, 100.0f },

            { 0,  4,  100.0f }, { 4,  8,  100.0f }, { 8,  12, 100.0f },
            { 1,  5,  100.0f }, { 5,  9,  100.0f }, { 9,  13, 100.0f },
            { 2,  6,  100.0f }, { 6,  10, 100.0f }, { 10, 14, 100.0f },
            { 3,  7,  100.0f }, { 7,  11, 100.0f }, { 11, 15, 100.0f },

            { 0,  5,  141.0f }, { 4,  9,  141.0f }, { 8,  13, 141.0f },
            { 1,  6,  141.0f }, { 5,  10, 141.0f }, { 9,  14, 141.0f },
            { 2,  7,  141.0f }, { 6,  11, 141.0f }, { 10, 15, 141.0f },

            { 1,  4,  141.0f }, { 5,  8,  141.0f }, { 9,  12, 141.0f },
            { 2,  5,  141.0f }, { 6,  9,  141.0f }, { 10, 13, 141.0f },
            { 3,  6,  141.0f }, { 7,  10, 141.0f }, { 11, 14, 141.0f },

            { 1,  0,  100.0f }, { 2,  1,  100.0f }, { 3,  2,  100.0f },
            { 5,  4,  100.0f }, { 6,  5,  100.0f }, { 7,  6,  100.0f },
            { 9,  8,  100.0f }, { 10, 9,  100.0f }, { 11, 10, 100.0f },
            { 13, 12, 100.0f }, { 14, 13, 100.0f }, { 15, 14, 100.0f },

            { 4,  0,  100.0f }, { 8,  4,  100.0f }, { 12, 8,  100.0f },
            { 5,  1,  100.0f }, { 9,  5,  100.0f }, { 13, 9,  100.0f },
            { 6,  2,  100.0f }, { 10, 6,  100.0f }, { 14, 10, 100.0f },
            { 7,  3,  100.0f }, { 11, 7,  100.0f }, { 15, 11, 100.0f },

            { 5,  0,  141.0f }, { 9,  4,  141.0f }, { 8,  13, 141.0f },
            { 6,  1,  141.0f }, { 10, 5,  141.0f }, { 9,  14, 141.0f },
            { 7,  2,  141.0f }, { 11, 6,  141.0f }, { 10, 15, 141.0f },

            { 4,  1,  141.0f }, { 8,  5,  141.0f }, { 12, 9,  141.0f },
            { 5,  2,  141.0f }, { 9,  6,  141.0f }, { 13, 10, 141.0f },
            { 6,  3,  141.0f }, { 10, 7,  141.0f }, { 14, 11, 141.0f },
        };

        if (!compute_path(vertices, edges, sizeof(vertices) / sizeof(VERTEX_T), sizeof(edges) / sizeof(EDGE_T)))
        {
            fprintf(stderr, "%s:%d:%s: Failed to compute path\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        print_path(vertices, 14);

        return true;
    }
}


int main()
{
    if (!test_1())
    {
        fprintf(stderr, "%s:%d:%s: Test failed\n", __FILE__, __LINE__, __FUNCTION__);
        return -1;
    }

    if (!test_2())
    {
        fprintf(stderr, "%s:%d:%s: Test failed\n", __FILE__, __LINE__, __FUNCTION__);
        return -1;
    }

    if (!test_3())
    {
        fprintf(stderr, "%s:%d:%s: Test failed\n", __FILE__, __LINE__, __FUNCTION__);
        return -1;
    }

    return 0;
}
