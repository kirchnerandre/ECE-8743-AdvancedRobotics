
#include <stdio.h>

#include "PolygonCpu.h"


namespace
{
    bool test_both_egdes_vertical()
    {
        VERTEX_T    vertices_a_100_200[]    = { { 100, 100 }, { 100, 200 } };
        VERTEX_T    vertices_a_200_400[]    = { { 100, 200 }, { 100, 400 } };
        VERTEX_T    vertices_a_400_500[]    = { { 100, 400 }, { 100, 500 } };
        VERTEX_T    vertices_a_500_700[]    = { { 100, 500 }, { 100, 700 } };
        VERTEX_T    vertices_a_700_800[]    = { { 100, 700 }, { 100, 800 } };

        VERTEX_T    vertices_b_100_200[]    = { { 200, 100 }, { 200, 200 } };
        VERTEX_T    vertices_b_200_400[]    = { { 200, 200 }, { 200, 400 } };
        VERTEX_T    vertices_b_400_500[]    = { { 200, 400 }, { 200, 500 } };
        VERTEX_T    vertices_b_500_700[]    = { { 200, 500 }, { 200, 700 } };
        VERTEX_T    vertices_b_700_800[]    = { { 200, 700 }, { 200, 800 } };

        EDGE_T      edge                    = { { 100, 300 }, { 100, 600 }, false };

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
        VERTEX_T    vertices_100_200[]  = { { 100, 200 }, { 100, 400 } };

        EDGE_T      edge_above          = { { 0,   0   }, { 200, 300 }, false };
        EDGE_T      edge_middle         = { { 0,   300 }, { 200, 300 }, false };
        EDGE_T      edge_bellow         = { { 0,   600 }, { 200, 300 }, false };

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
        VERTEX_T    vertices_100_200_above []   = { { 0,   300 }, { 200, 0   } };
        VERTEX_T    vertices_100_200_middle[]   = { { 0,   300 }, { 200, 300 } };
        VERTEX_T    vertices_100_200_bellow[]   = { { 0,   300 }, { 200, 600 } };

        EDGE_T      edge                        = { { 100, 200 }, { 100, 400 }, false };

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


    bool test_none_egde_vertical()
    {
        VERTEX_T    vertices_ab[]       = { { 100, 100 }, { 200, 200 } };

        EDGE_T      edge_cd_through     = { { 200, 100 }, { 100, 200 }, false };
        EDGE_T      edge_cd_through_not = { { 300, 100 }, { 400, 200 }, false };

        if (test_colision(vertices_ab, 0, 2, edge_cd_through_not) != false)
        {
            fprintf(stderr, "%s:%d:%s: Incorrect collision test output\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        if (test_colision(vertices_ab, 0, 2, edge_cd_through) != true)
        {
            fprintf(stderr, "%s:%d:%s: Incorrect collision test output\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        return true;
    }


    bool test_edge_inside_polygon()
    {
        VERTEX_T    vertices[]  = { { 100, 100 }, { 200, 100 }, { 200, 200 }, { 100, 200 },
                                    { 500, 100 }, { 600, 100 }, { 600, 200 }, { 500, 200 } };

        EDGE_T      edge_1      = { { 10,  10  }, { 20,  20  }, false };
        EDGE_T      edge_2      = { { 110, 150 }, { 120, 160 }, false };
        EDGE_T      edge_3      = { { 180, 110 }, { 190, 120 }, false };
        EDGE_T      edge_4      = { { 510, 150 }, { 520, 150 }, false };
        EDGE_T      edge_5      = { { 580, 110 }, { 590, 120 }, false };

        if (test_colision(vertices, 0, 4, edge_1) != false)
        {
            fprintf(stderr, "%s:%d:%s: Incorrect collision test output\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        if (test_colision(vertices, 0, 4, edge_2) != true)
        {
            fprintf(stderr, "%s:%d:%s: Incorrect collision test output\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        if (test_colision(vertices, 0, 4, edge_3) != true)
        {
            fprintf(stderr, "%s:%d:%s: Incorrect collision test output\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        if (test_colision(vertices, 0, 4, edge_4) != false)
        {
            fprintf(stderr, "%s:%d:%s: Incorrect collision test output\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        if (test_colision(vertices, 0, 4, edge_5) != false)
        {
            fprintf(stderr, "%s:%d:%s: Incorrect collision test output\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        if (test_colision(vertices, 4, 8, edge_1) != false)
        {
            fprintf(stderr, "%s:%d:%s: Incorrect collision test output\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        if (test_colision(vertices, 4, 8, edge_2) != false)
        {
            fprintf(stderr, "%s:%d:%s: Incorrect collision test output\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        if (test_colision(vertices, 4, 8, edge_3) != false)
        {
            fprintf(stderr, "%s:%d:%s: Incorrect collision test output\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        if (test_colision(vertices, 4, 8, edge_4) != true)
        {
            fprintf(stderr, "%s:%d:%s: Incorrect collision test output\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        if (test_colision(vertices, 4, 8, edge_5) != true)
        {
            fprintf(stderr, "%s:%d:%s: Incorrect collision test output\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        return true;
    }
}


int main(int argc, char** argv)
{
    if (!test_both_egdes_vertical())
    {
        fprintf(stderr, "%s:%d:%s: Test failed\n", __FILE__, __LINE__, __FUNCTION__);
        return -1;
    }

    if (!test_first_egde_vertical())
    {
        fprintf(stderr, "%s:%d:%s: Test failed\n", __FILE__, __LINE__, __FUNCTION__);
        return -1;
    }

    if (!test_second_egde_vertical())
    {
        fprintf(stderr, "%s:%d:%s: Test failed\n", __FILE__, __LINE__, __FUNCTION__);
        return -1;
    }

    if (!test_none_egde_vertical())
    {
        fprintf(stderr, "%s:%d:%s: Test failed\n", __FILE__, __LINE__, __FUNCTION__);
        return -1;
    }

    if (!test_edge_inside_polygon())
    {
        fprintf(stderr, "%s:%d:%s: Test failed\n", __FILE__, __LINE__, __FUNCTION__);
        return -1;
    }

    return 0;
}
