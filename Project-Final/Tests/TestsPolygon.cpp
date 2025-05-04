
#include <stdio.h>

#include "PolygonCpu.h"


namespace
{
    bool test_both_egdes_vertical()
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


    bool test_first_egde_vertical()
    {
        VERTICES_T  vertices_100_200  = { { 100, 200 }, { 100, 400 } };

        EDGE_T      edge_above        = { { 0,   0   }, { 200, 300 }, false };
        EDGE_T      edge_middle       = { { 0,   300 }, { 200, 300 }, false };
        EDGE_T      edge_bellow       = { { 0,   600 }, { 200, 300 }, false };

        if (test_colision(vertices_100_200, 0, 2, edge_above) != false)
        {
            fprintf(stderr, "%s:%d:%s: Incorrect collision test output\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        if (test_colision(vertices_100_200, 0, 2, edge_middle) != true)
        {
            fprintf(stderr, "%s:%d:%s: Incorrect collision test output\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        if (test_colision(vertices_100_200, 0, 2, edge_bellow) != false)
        {
            fprintf(stderr, "%s:%d:%s: Incorrect collision test output\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }
        return true;
    }


    bool test_second_egde_vertical()
    {
        VERTICES_T  vertices_100_200_above  = { { 0,   300 }, { 200, 0   } };
        VERTICES_T  vertices_100_200_middle = { { 0,   300 }, { 200, 300 } };
        VERTICES_T  vertices_100_200_bellow = { { 0,   300 }, { 200, 600 } };

        EDGE_T      edge                    = { { 100, 200 }, { 100, 400 }, false };

        if (test_colision(vertices_100_200_above, 0, 2, edge) != false)
        {
            fprintf(stderr, "%s:%d:%s: Incorrect collision test output\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        if (test_colision(vertices_100_200_middle, 0, 2, edge) != true)
        {
            fprintf(stderr, "%s:%d:%s: Incorrect collision test output\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        if (test_colision(vertices_100_200_bellow, 0, 2, edge) != false)
        {
            fprintf(stderr, "%s:%d:%s: Incorrect collision test output\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        return true;
    }
}


int main(int argc, char** argv)
{
#if 0
    if (!test_both_egdes_vertical())
    {
        fprintf(stderr, "%s:%d:%s: Test failed\n", __FILE__, __LINE__, __FUNCTION__);
        return -1;
    }
#endif
#if 0
    if (!test_first_egde_vertical())
    {
        fprintf(stderr, "%s:%d:%s: Test failed\n", __FILE__, __LINE__, __FUNCTION__);
        return -1;
    }
#endif
#if 1
    if (!test_second_egde_vertical())
    {
        fprintf(stderr, "%s:%d:%s: Test failed\n", __FILE__, __LINE__, __FUNCTION__);
        return -1;
    }
#endif
    printf("Helloe owlrd\n");

    return 0;
}
