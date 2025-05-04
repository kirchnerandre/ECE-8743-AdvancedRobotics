
#include <stdio.h>

#include "PolygonCpu.h"


namespace
{
    bool test_egdes_vertical()
    {
        VERTICES_T  vertices_a_100_200  = { { 100, 100 }, { 100, 200 } };
        VERTICES_T  vertices_a_200_400  = { { 100, 200 }, { 100, 400 } };
        VERTICES_T  vertices_a_400_500  = { { 100, 400 }, { 100, 500 } };
        VERTICES_T  vertices_a_500_700  = { { 100, 500 }, { 100, 700 } };
        VERTICES_T  vertices_a_700_800  = { { 100, 700 }, { 100, 800 } };

        VERTICES_T  vertices_b_100_200  = { { 200, 100 }, { 200, 200 } };
        VERTICES_T  vertices_b_200_400  = { { 200, 200 }, { 200, 400 } };
        VERTICES_T  vertices_b_400_500  = { { 200, 400 }, { 200, 500 } };
        VERTICES_T  vertices_b_500_700  = { { 200, 500 }, { 200, 700 } };
        VERTICES_T  vertices_b_700_800  = { { 200, 700 }, { 200, 800 } };

        EDGE_T      edge                = { { 100, 300 }, { 100, 600 }, false };

        bool        expected_1[]        = { false,
                                            true,
                                            true,
                                            true,
                                            false };

        bool        expected_2[]        = { false,
                                            false,
                                            false,
                                            false,
                                            false };
#if 0
        if (test_colision(vertices_a_100_200, 0, 2, edge) != false)
        {
            fprintf(stderr, "%s:%d:%s: Incorrect collision test output\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        if (test_colision(vertices_a_200_400, 0, 2, edge) != true)
        {
            fprintf(stderr, "%s:%d:%s: Incorrect collision test output\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        if (test_colision(vertices_a_400_500, 0, 2, edge) != true)
        {
            fprintf(stderr, "%s:%d:%s: Incorrect collision test output\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        if (test_colision(vertices_a_500_700, 0, 2, edge) != true)
        {
            fprintf(stderr, "%s:%d:%s: Incorrect collision test output\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        if (test_colision(vertices_a_700_800, 0, 2, edge) != false)
        {
            fprintf(stderr, "%s:%d:%s: Incorrect collision test output\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }
#endif
        if (test_colision(vertices_b_100_200, 0, 2, edge) != false)
        {
            fprintf(stderr, "%s:%d:%s: Incorrect collision test output\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        if (test_colision(vertices_b_200_400, 0, 2, edge) != false)
        {
            fprintf(stderr, "%s:%d:%s: Incorrect collision test output\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        if (test_colision(vertices_b_400_500, 0, 2, edge) != false)
        {
            fprintf(stderr, "%s:%d:%s: Incorrect collision test output\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        if (test_colision(vertices_b_500_700, 0, 2, edge) != false)
        {
            fprintf(stderr, "%s:%d:%s: Incorrect collision test output\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        if (test_colision(vertices_b_700_800, 0, 2, edge) != false)
        {
            fprintf(stderr, "%s:%d:%s: Incorrect collision test output\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        return true;
    }
}


int main(int argc, char** argv)
{
    if (!test_egdes_vertical())
    {
        fprintf(stderr, "%s:%d:%s: Test failed\n", __FILE__, __LINE__, __FUNCTION__);
        return -1;
    }

    printf("Helloe owlrd\n");

    return 0;
}
