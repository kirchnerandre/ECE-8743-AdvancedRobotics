
#include "DijkstraGpu.h"


namespace
{
    void print_path(VERTEX_T* vertices, size_t VerticesSize, int32_t VertexEnd)
    {
        int32_t source = VertexEnd;

        while (1)
        {
            if (source < 0)
            {
                break;
            }
            else
            {
                printf("%d\n", source);
            }

            source = vertices[source].Source;
        }
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

        print_path(vertices, sizeof(vertices) / sizeof(VERTEX_T), 3);

        return true;
    }


    bool test_2()
    {
        VERTEX_T    vertices[]  = {
            { 100, 100, false }, { 200, 100, false }, { 300, 100, false }, { 400, 100, false },
            { 100, 200, false }, { 200, 200, false }, { 300, 200, false }, { 400, 200, false },
            { 100, 300, false }, { 200, 300, false }, { 300, 300, false }, { 400, 300, false },
            { 100, 400, false }, { 200, 400, false }, { 300, 400, false }, { 400, 400, false } };

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
        };

        if (!compute_path(vertices, edges, sizeof(vertices) / sizeof(VERTEX_T), sizeof(edges) / sizeof(EDGE_T)))
        {
            fprintf(stderr, "%s:%d:%s: Failed to compute path\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

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

    return 0;
}
