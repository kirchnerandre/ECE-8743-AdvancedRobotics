
#include "DijkstraGpu.h"


namespace
{
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
            { 100, 100, -1, true,   0.0f }, { 200, 100, -1, false, -1.0f },
            { 100, 200, -1, false, -1.0f }, { 200, 200, -1, false, -1.0f } };

        EDGE_T      edges[]     = {
            { 0,  1,  false, 100.0f },
            { 1,  2,  false, 141.0f },
            { 2,  3,  false, 100.0f },
            { 3,  0,  false, 141.0f },
            { 0,  2,  false, 100.0f },
            { 1,  3,  false, 100.0f },

            { 1,  0,  false, 100.0f },
            { 2,  1,  false, 141.0f },
            { 3,  2,  false, 100.0f },
            { 0,  3,  false, 141.0f },
            { 2,  0,  false, 100.0f },
            { 3,  1,  false, 100.0f },
        };

        if (!compute_path(vertices, edges, sizeof(vertices) / sizeof(VERTEX_T), sizeof(edges) / sizeof(EDGE_T)))
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
            { 100, 100, -1, true,   0.0f }, { 200, 100, -1, false, -1.0f }, { 300, 100, -1, false, -1.0f },
            { 100, 200, -1, false, -1.0f }, { 200, 200, -1, false, -1.0f }, { 300, 200, -1, false, -1.0f },
            { 100, 300, -1, false, -1.0f }, { 200, 300, -1, false, -1.0f }, { 300, 300, -1, false, -1.0f } };

        EDGE_T      edges[]     = {
            { 0,  1,  false, 100.0f }, { 1,  2,  false, 100.0f },
            { 3,  4,  false, 100.0f }, { 4,  5,  false, 100.0f },
            { 6,  7,  false, 100.0f }, { 7,  8,  false, 100.0f },

            { 0,  3,  false, 100.0f }, { 1,  4,  false, 100.0f }, { 2,  5,  false, 100.0f },
            { 3,  6,  false, 100.0f }, { 4,  7,  false, 100.0f }, { 5,  8,  false, 100.0f },

            { 0,  4,  false, 141.0f }, { 1,  5,  false, 141.0f },
            { 3,  7,  false, 141.0f }, { 4,  8,  false, 141.0f },

            { 3,  1,  false, 141.0f }, { 4,  2,  false, 141.0f },
            { 6,  4,  false, 141.0f }, { 7,  5,  false, 141.0f },
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
            { 100, 100, -1, true,   0.0f }, { 200, 100, -1, false, -1.0f }, { 300, 100, -1, false, -1.0f }, { 400, 100, -1, false, -1.0f },
            { 100, 200, -1, false, -1.0f }, { 200, 200, -1, false, -1.0f }, { 300, 200, -1, false, -1.0f }, { 400, 200, -1, false, -1.0f },
            { 100, 300, -1, false, -1.0f }, { 200, 300, -1, false, -1.0f }, { 300, 300, -1, false, -1.0f }, { 400, 300, -1, false, -1.0f },
            { 100, 400, -1, false, -1.0f }, { 200, 400, -1, false, -1.0f }, { 300, 400, -1, false, -1.0f }, { 400, 400, -1, false, -1.0f } };

        EDGE_T      edges[]     = {
            { 0,  1,  false, 100.0f }, { 1,  2,  false, 100.0f }, { 2,  3,  false, 100.0f },
            { 4,  5,  false, 100.0f }, { 5,  6,  false, 100.0f }, { 6,  7,  false, 100.0f },
            { 8,  9,  false, 100.0f }, { 9,  10, false, 100.0f }, { 10, 11, false, 100.0f },
            { 12, 13, false, 100.0f }, { 13, 14, false, 100.0f }, { 14, 15, false, 100.0f },

            { 0,  4,  false, 100.0f }, { 4,  8,  false, 100.0f }, { 8,  12, false, 100.0f },
            { 1,  5,  false, 100.0f }, { 5,  9,  false, 100.0f }, { 9,  13, false, 100.0f },
            { 2,  6,  false, 100.0f }, { 6,  10, false, 100.0f }, { 10, 14, false, 100.0f },
            { 3,  7,  false, 100.0f }, { 7,  11, false, 100.0f }, { 11, 15, false, 100.0f },

            { 0,  5,  false, 141.0f }, { 4,  9,  false, 141.0f }, { 8,  13, false, 141.0f },
            { 1,  6,  false, 141.0f }, { 5,  10, false, 141.0f }, { 9,  14, false, 141.0f },
            { 2,  7,  false, 141.0f }, { 6,  11, false, 141.0f }, { 10, 15, false, 141.0f },

            { 1,  4,  false, 141.0f }, { 5,  8,  false, 141.0f }, { 9,  12, false, 141.0f },
            { 2,  5,  false, 141.0f }, { 6,  9,  false, 141.0f }, { 10, 13, false, 141.0f },
            { 3,  6,  false, 141.0f }, { 7,  10, false, 141.0f }, { 11, 14, false, 141.0f },

            { 1,  0,  false, 100.0f }, { 2,  1,  false, 100.0f }, { 3,  2,  false, 100.0f },
            { 5,  4,  false, 100.0f }, { 6,  5,  false, 100.0f }, { 7,  6,  false, 100.0f },
            { 9,  8,  false, 100.0f }, { 10, 9,  false, 100.0f }, { 11, 10, false, 100.0f },
            { 13, 12, false, 100.0f }, { 14, 13, false, 100.0f }, { 15, 14, false, 100.0f },

            { 4,  0,  false, 100.0f }, { 8,  4,  false, 100.0f }, { 12, 8,  false, 100.0f },
            { 5,  1,  false, 100.0f }, { 9,  5,  false, 100.0f }, { 13, 9,  false, 100.0f },
            { 6,  2,  false, 100.0f }, { 10, 6,  false, 100.0f }, { 14, 10, false, 100.0f },
            { 7,  3,  false, 100.0f }, { 11, 7,  false, 100.0f }, { 15, 11, false, 100.0f },

            { 5,  0,  false, 141.0f }, { 9,  4,  false, 141.0f }, { 8,  13, false, 141.0f },
            { 6,  1,  false, 141.0f }, { 10, 5,  false, 141.0f }, { 9,  14, false, 141.0f },
            { 7,  2,  false, 141.0f }, { 11, 6,  false, 141.0f }, { 10, 15, false, 141.0f },

            { 4,  1,  false, 141.0f }, { 8,  5,  false, 141.0f }, { 12, 9,  false, 141.0f },
            { 5,  2,  false, 141.0f }, { 9,  6,  false, 141.0f }, { 13, 10, false, 141.0f },
            { 6,  3,  false, 141.0f }, { 10, 7,  false, 141.0f }, { 14, 11, false, 141.0f },
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
