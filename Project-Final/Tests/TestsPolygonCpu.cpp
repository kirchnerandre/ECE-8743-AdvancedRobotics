
#include <stdio.h>

#include "PolygonCpu.h"


namespace
{
    bool test_both_egdes_vertical()
    {
        VERTEX_T    vertices[]  = {
            { 100.0f, 100.0f }, { 100.0f, 200.0f }, { 100.0f, 400.0f }, { 100.0f, 500.0f }, { 100.0f, 700.0f }, { 100.0f, 800.0f },
            { 200.0f, 100.0f }, { 200.0f, 200.0f }, { 200.0f, 400.0f }, { 200.0f, 500.0f }, { 200.0f, 700.0f }, { 200.0f, 800.0f },
            { 100.0f, 300.0f }, { 100.0f, 600.0f },
        };

        EDGE_T      edge        = { 12, 13, false };

        if (test_colision(vertices, 0, 2, edge) != false)
        {
            fprintf(stderr, "%s:%d:%s: Incorrect collision test output\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        if (test_colision(vertices, 1, 3, edge) != true)
        {
            fprintf(stderr, "%s:%d:%s: Incorrect collision test output\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        if (test_colision(vertices, 2, 4, edge) != true)
        {
            fprintf(stderr, "%s:%d:%s: Incorrect collision test output\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        if (test_colision(vertices, 3, 5, edge) != true)
        {
            fprintf(stderr, "%s:%d:%s: Incorrect collision test output\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        if (test_colision(vertices, 4, 6, edge) != false)
        {
            fprintf(stderr, "%s:%d:%s: Incorrect collision test output\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        if (test_colision(vertices, 6, 8, edge) != false)
        {
            fprintf(stderr, "%s:%d:%s: Incorrect collision test output\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        if (test_colision(vertices, 7, 9, edge) != false)
        {
            fprintf(stderr, "%s:%d:%s: Incorrect collision test output\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        if (test_colision(vertices, 8, 10, edge) != false)
        {
            fprintf(stderr, "%s:%d:%s: Incorrect collision test output\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        if (test_colision(vertices, 9, 11, edge) != false)
        {
            fprintf(stderr, "%s:%d:%s: Incorrect collision test output\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        if (test_colision(vertices, 10, 12, edge) != false)
        {
            fprintf(stderr, "%s:%d:%s: Incorrect collision test output\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        return true;
    }


    bool test_first_egde_vertical()
    {
        VERTEX_T    vertices[] = {
            { 100.0f, 200.0f },
            { 100.0f, 400.0f },
            { 0.0f,   0.0f   },
            { 0.0f,   300.0f },
            { 0.0f,   600.0f },
            { 200.0f, 300.0f }
        };

        EDGE_T      edge_above          = { 2, 5, false };
        EDGE_T      edge_middle         = { 3, 5, false };
        EDGE_T      edge_bellow         = { 4, 5, false };

        if (test_colision(vertices, 0, 2, edge_above) != false)
        {
            fprintf(stderr, "%s:%d:%s: Incorrect collision test output\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        if (test_colision(vertices, 0, 2, edge_middle) != true)
        {
            fprintf(stderr, "%s:%d:%s: Incorrect collision test output\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        if (test_colision(vertices, 0, 2, edge_bellow) != false)
        {
            fprintf(stderr, "%s:%d:%s: Incorrect collision test output\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        return true;
    }


    bool test_second_egde_vertical()
    {
        VERTEX_T    vertices_100_200_above []   = { { 0.0f,   300.0f }, { 200.0f, 0.0f   }, { 100.0f, 200.0f }, { 100.0f, 400.0f } };
        VERTEX_T    vertices_100_200_middle[]   = { { 0.0f,   300.0f }, { 200.0f, 300.0f }, { 100.0f, 200.0f }, { 100.0f, 400.0f } };
        VERTEX_T    vertices_100_200_bellow[]   = { { 0.0f,   300.0f }, { 200.0f, 600.0f }, { 100.0f, 200.0f }, { 100.0f, 400.0f } };

        EDGE_T      edge                        = { 2, 3, false };

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
        VERTEX_T    vertices[]          = { { 100.0f, 100.0f },
                                            { 200.0f, 200.0f },
                                            { 200.0f, 100.0f },
                                            { 100.0f, 200.0f },
                                            { 300.0f, 100.0f },
                                            { 400.0f, 200.0f } };

        EDGE_T      edge_cd_through     = { 2, 3, false };
        EDGE_T      edge_cd_through_not = { 4, 5, false };

        if (test_colision(vertices, 0, 2, edge_cd_through_not) != false)
        {
            fprintf(stderr, "%s:%d:%s: Incorrect collision test output\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        if (test_colision(vertices, 0, 2, edge_cd_through) != true)
        {
            fprintf(stderr, "%s:%d:%s: Incorrect collision test output\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        return true;
    }


    bool test_edge_inside_polygon()
    {
        VERTEX_T    vertices[]  = { { 100, 100 }, { 200, 100 }, { 200, 200 }, { 100, 200 },
                                    { 500, 100 }, { 600, 100 }, { 600, 200 }, { 500, 200 },
                                    { 10,  10  }, { 20,  20  },
                                    { 110, 150 }, { 120, 160 },
                                    { 180, 110 }, { 190, 120 },
                                    { 510, 150 }, { 520, 150 },
                                    { 580, 110 }, { 590, 120 } };

        EDGE_T      edge_1      = { 8,  9,  false };
        EDGE_T      edge_2      = { 10, 11, false };
        EDGE_T      edge_3      = { 12, 13, false };
        EDGE_T      edge_4      = { 14, 15, false };
        EDGE_T      edge_5      = { 16, 17, false };

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
