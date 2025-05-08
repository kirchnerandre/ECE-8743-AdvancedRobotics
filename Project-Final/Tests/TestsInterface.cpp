
#include "DataTypes.h"
#include "Interface.h"


namespace
{
    bool compare_vertices(VERTICES_T& VerticesA, VERTICES_T& VerticesB)
    {
        if (VerticesA.size() != VerticesB.size())
        {
            fprintf(stderr, "%s:%d:%s: Different vertices size\n", __FILE__, __LINE__, __FUNCTION__);
            return false, 0.0f;
        }

        for (size_t i = 0u; i < VerticesA.size(); i++)
        {
            if ((VerticesA[i].X != VerticesB[i].X) || (VerticesA[i].Y != VerticesB[i].Y))
            {
                fprintf(stderr, "%s:%d:%s: Different vertices values\n", __FILE__, __LINE__, __FUNCTION__);
                return false, 0.0f;
            }
        }

        return true;
    }


    bool compare_edges(EDGES_T& EdgesA, EDGES_T& EdgesB)
    {
        if (EdgesA.size() != EdgesB.size())
        {
            fprintf(stderr, "%s:%d:%s: Different edges size\n", __FILE__, __LINE__, __FUNCTION__);
            return false, 0.0f;
        }

        for (size_t i = 0u; i < EdgesA.size(); i++)
        {
            if ((EdgesA[i].IndexA != EdgesB[i].IndexA)
             || (EdgesA[i].IndexB != EdgesB[i].IndexB)
             || (EdgesA[i].Status != EdgesB[i].Status))
            {
                fprintf(stderr, "%s:%d:%s: Different edges values\n", __FILE__, __LINE__, __FUNCTION__);
                return false, 0.0f;
            }
        }

        return true;
    }


    bool compare_numbers(NUMBERS_T& NumbersA, NUMBERS_T& NumbersB)
    {
        if (NumbersA.size() != NumbersB.size())
        {
            fprintf(stderr, "%s:%d:%s: Different numbers size\n", __FILE__, __LINE__, __FUNCTION__);
            return false, 0.0f;
        }

        for (size_t i = 0u; i < NumbersA.size(); i++)
        {
            if (NumbersA[i] != NumbersB[i])
            {
                fprintf(stderr, "%s:%d:%s: Different numbers values\n", __FILE__, __LINE__, __FUNCTION__);
                return false, 0.0f;
            }
        }

        return true;
    }


    bool test(
        std::string&    Filename,
        VERTICES_T&     ExpectedVertices,
        EDGES_T&        ExpectedEdges,
        NUMBERS_T&      ExpectedPolygonsBegin,
        NUMBERS_T&      ExpectedPolygonsEnd)
    {
        VERTICES_T  vertices;
        EDGES_T     edges;
        NUMBERS_T   polygons_begin;
        NUMBERS_T   polygons_end;

        if(!read_map(vertices, edges, polygons_begin, polygons_end, Filename))
        {
            fprintf(stderr, "%s:%d:%s: Failed to read polygons\n", __FILE__, __LINE__, __FUNCTION__);
            return false, 0.0f;
        }

        if (!compare_vertices(vertices, ExpectedVertices))
        {
            fprintf(stderr, "%s:%d:%s: Invalid vertices\n", __FILE__, __LINE__, __FUNCTION__);
            return false, 0.0f;
        }

        if (!compare_edges(edges, ExpectedEdges))
        {
            fprintf(stderr, "%s:%d:%s: Invalid edges\n", __FILE__, __LINE__, __FUNCTION__);
            return false, 0.0f;
        }

        if (!compare_numbers(polygons_begin, ExpectedPolygonsBegin))
        {
            fprintf(stderr, "%s:%d:%s: Invalid polygons begin\n", __FILE__, __LINE__, __FUNCTION__);
            return false, 0.0f;
        }

        if (!compare_numbers(polygons_end, ExpectedPolygonsEnd))
        {
            fprintf(stderr, "%s:%d:%s: Invalid polygons end\n", __FILE__, __LINE__, __FUNCTION__);
            return false, 0.0f;
        }

        return true;
    }
}


int main()
{
    std::string         filename_1          = "../../Data/Maps/Map1.txt";
    std::string         filename_2          = "../../Data/Maps/Map2.txt";
    std::string         filename_3          = "../../Data/Maps/Map3.txt";

    const size_t        expected_polygons_1 = 7u;
    const size_t        expected_polygons_2 = 15u;
    const size_t        expected_polygons_3 = 5u;

    VERTICES_T  expected_vertices_1         = {
        {    2.0f,    8.0f, false }, {    2.0f,   16.0f, false }, {    4.0f,   16.0f, false }, {    4.0f,    8.0f, false },
        {    8.0f,   13.0f, false }, {    8.0f,   16.0f, false }, {    9.0f,   16.0f, false }, {    9.0f,   13.0f, false },
        {   12.0f,   11.0f, false }, {   12.0f,   14.0f, false }, {   15.0f,   14.0f, false }, {   15.0f,   11.0f, false },
        {    7.0f,    6.0f, false }, {    7.0f,   10.0f, false }, {    9.0f,   10.0f, false }, {    9.0f,    6.0f, false },
        {    1.0f,    3.0f, false }, {    1.0f,    6.0f, false }, {    4.0f,    6.0f, false }, {    4.0f,    3.0f, false },
        {    5.0f,    1.0f, false }, {    5.0f,    2.0f, false }, {    9.0f,    2.0f, false }, {    9.0f,    1.0f, false },
        {   11.0f,    0.0f, false }, {   11.0f,    8.0f, false }, {   13.0f,    8.0f, false }, {   13.0f,    0.0f, false }, };

    VERTICES_T  expected_vertices_2         = {
        {  600.0f, 1800.0f, false }, {  650.0f, 1900.0f, false }, {  750.0f, 1850.0f, false }, {  850.0f, 1650.0f, false },
        {  100.0f,  600.0f, false }, {  180.0f,  700.0f, false }, {  300.0f,  600.0f, false }, {  220.0f,  450.0f, false },
        {  200.0f,  850.0f, false }, {  250.0f, 1100.0f, false }, {  350.0f, 1150.0f, false }, {  350.0f,  750.0f, false },
        {  200.0f, 1250.0f, false }, {  100.0f, 1450.0f, false }, {  250.0f, 1800.0f, false }, {  400.0f, 1500.0f, false }, {  350.0f, 1200.0f, false },
        {  450.0f,   50.0f, false }, {  350.0f,  200.0f, false }, {  500.0f,  350.0f, false }, {  900.0f,   50.0f, false },
        {  600.0f,  350.0f, false }, {  550.0f,  500.0f, false }, {  550.0f,  600.0f, false }, {  700.0f,  600.0f, false }, {  850.0f,  500.0f, false },
        {  450.0f,  850.0f, false }, {  500.0f, 1100.0f, false }, {  600.0f, 1200.0f, false }, {  550.0f,  750.0f, false },
        {  900.0f,  450.0f, false }, {  850.0f,  650.0f, false }, { 1100.0f,  700.0f, false }, { 1200.0f,  500.0f, false },
        {  850.0f,  950.0f, false }, {  800.0f, 1300.0f, false }, {  900.0f, 1350.0f, false }, { 1100.0f, 1200.0f, false }, { 1100.0f,  900.0f, false },
        { 1250.0f,  600.0f, false }, { 1300.0f,  700.0f, false }, { 1400.0f,  650.0f, false }, { 1500.0f,  500.0f, false }, { 1350.0f,  350.0f, false },
        { 1150.0f, 1200.0f, false }, { 1300.0f, 1250.0f, false }, { 1400.0f, 1000.0f, false }, { 1200.0f,  950.0f, false },
        { 1150.0f, 1550.0f, false }, { 1050.0f, 1800.0f, false }, { 1100.0f, 1850.0f, false }, { 1300.0f, 1750.0f, false }, { 1250.0f, 1500.0f, false },
        { 1500.0f,  900.0f, false }, { 1600.0f,  950.0f, false }, { 1750.0f,  750.0f, false }, { 1700.0f,  700.0f, false },
        { 1400.0f, 1300.0f, false }, { 1350.0f, 1650.0f, false }, { 1450.0f, 1700.0f, false }, { 1650.0f, 1400.0f, false },
        { 1650.0f, 1050.0f, false }, { 1700.0f, 1300.0f, false }, { 1950.0f, 1200.0f, false }, { 1900.0f,  950.0f, false }, };

    VERTICES_T  expected_vertices_3         = {
        {   10.0f,  140.0f, false }, {   10.0f,  150.0f, false }, {   20.0f,  150.0f, false }, {   20.0f,  170.0f, false }, {   30.0f,  170.0f, false },
        {   30.0f,  180.0f, false }, {   40.0f,  180.0f, false }, {   40.0f,  190.0f, false }, {   50.0f,  190.0f, false }, {   50.0f,  200.0f, false },
        {   60.0f,  200.0f, false }, {   60.0f,  190.0f, false }, {   70.0f,  190.0f, false }, {   70.0f,  180.0f, false }, {   80.0f,  180.0f, false },
        {   80.0f,  150.0f, false }, {   70.0f,  150.0f, false }, {   70.0f,  140.0f, false }, {   60.0f,  140.0f, false }, {   60.0f,  130.0f, false },
        {   40.0f,  130.0f, false }, {   40.0f,  120.0f, false }, {   30.0f,  120.0f, false }, {   30.0f,  130.0f, false }, {   20.0f,  130.0f, false },
        {   20.0f,  140.0f, false },

        {   90.0f,  140.0f, false }, {   90.0f,  150.0f, false }, {  100.0f,  150.0f, false }, {  100.0f,  180.0f, false }, {  120.0f,  180.0f, false },
        {  120.0f,  190.0f, false }, {  130.0f,  190.0f, false }, {  130.0f,  180.0f, false }, {  150.0f,  180.0f, false }, {  150.0f,  170.0f, false },
        {  160.0f,  170.0f, false }, {  160.0f,  160.0f, false }, {  150.0f,  160.0f, false }, {  150.0f,  140.0f, false }, {  130.0f,  140.0f, false },
        {  130.0f,  130.0f, false }, {  110.0f,  130.0f, false }, {  110.0f,  120.0f, false }, {  100.0f,  120.0f, false }, {  100.0f,  140.0f, false },

        {  160.0f,  100.0f, false }, {  160.0f,  120.0f, false }, {  170.0f,  120.0f, false }, {  170.0f,  130.0f, false }, {  180.0f,  130.0f, false },
        {  180.0f,  140.0f, false }, {  190.0f,  140.0f, false }, {  190.0f,  150.0f, false }, {  210.0f,  150.0f, false }, {  210.0f,  160.0f, false },
        {  220.0f,  160.0f, false }, {  220.0f,  140.0f, false }, {  230.0f,  140.0f, false }, {  230.0f,  130.0f, false }, {  220.0f,  130.0f, false },
        {  220.0f,  100.0f, false }, {  210.0f,  100.0f, false }, {  210.0f,   90.0f, false }, {  180.0f,   90.0f, false }, {  180.0f,  100.0f, false },

        {  190.0f,  200.0f, false }, {  190.0f,  210.0f, false }, {  200.0f,  210.0f, false }, {  200.0f,  230.0f, false }, {  220.0f,  230.0f, false },
        {  220.0f,  220.0f, false }, {  230.0f,  220.0f, false }, {  230.0f,  190.0f, false }, {  200.0f,  190.0f, false }, {  200.0f,  200.0f, false },

        {   70.0f,   50.0f, false }, {   70.0f,   70.0f, false }, {   80.0f,   70.0f, false }, {   80.0f,   80.0f, false }, {  130.0f,   80.0f, false },
        {  130.0f,   70.0f, false }, {  150.0f,   70.0f, false }, {  150.0f,   60.0f, false }, {  160.0f,   60.0f, false }, {  160.0f,   50.0f, false },
        {  220.0f,   50.0f, false }, {  220.0f,   40.0f, false }, {  200.0f,   40.0f, false }, {  200.0f,   30.0f, false }, {  220.0f,   30.0f, false },
        {  220.0f,   20.0f, false }, {  180.0f,   20.0f, false }, {  180.0f,   10.0f, false }, {  170.0f,   10.0f, false }, {  170.0f,   20.0f, false },
        {  160.0f,   20.0f, false }, {  160.0f,   30.0f, false }, {  150.0f,   30.0f, false }, {  150.0f,   40.0f, false }, {   80.0f,   40.0f, false },
        {   80.0f,   50.0f } };

    EDGES_T     expected_edges_1            = {
        {  3,   0,  false, 0.0f }, {  0,   1,  false, 0.0f }, {  1,   2,  false, 0.0f }, {  2,   3,  false, 0.0f },
        {  7,   4,  false, 0.0f }, {  4,   5,  false, 0.0f }, {  5,   6,  false, 0.0f }, {  6,   7,  false, 0.0f },
        { 11,   8,  false, 0.0f }, {  8,   9,  false, 0.0f }, {  9,  10,  false, 0.0f }, { 10,  11,  false, 0.0f },
        { 15,  12,  false, 0.0f }, { 12,  13,  false, 0.0f }, { 13,  14,  false, 0.0f }, { 14,  15,  false, 0.0f },
        { 19,  16,  false, 0.0f }, { 16,  17,  false, 0.0f }, { 17,  18,  false, 0.0f }, { 18,  19,  false, 0.0f },
        { 23,  20,  false, 0.0f }, { 20,  21,  false, 0.0f }, { 21,  22,  false, 0.0f }, { 22,  23,  false, 0.0f },
        { 27,  24,  false, 0.0f }, { 24,  25,  false, 0.0f }, { 25,  26,  false, 0.0f }, { 26,  27,  false, 0.0f },
    };

    EDGES_T     expected_edges_2            = {
        {  3,   0,  false, 0.0f }, {  0,   1,  false, 0.0f }, {  1,   2,  false, 0.0f }, {  2,   3,  false, 0.0f },
        {  7,   4,  false, 0.0f }, {  4,   5,  false, 0.0f }, {  5,   6,  false, 0.0f }, {  6,   7,  false, 0.0f },
        { 11,   8,  false, 0.0f }, {  8,   9,  false, 0.0f }, {  9,  10,  false, 0.0f }, { 10,  11,  false, 0.0f },
        { 16,  12,  false, 0.0f }, { 12,  13,  false, 0.0f }, { 13,  14,  false, 0.0f }, { 14,  15,  false, 0.0f }, { 15,  16,  false, 0.0f },
        { 20,  17,  false, 0.0f }, { 17,  18,  false, 0.0f }, { 18,  19,  false, 0.0f }, { 19,  20,  false, 0.0f },
        { 25,  21,  false, 0.0f }, { 21,  22,  false, 0.0f }, { 22,  23,  false, 0.0f }, { 23,  24,  false, 0.0f }, { 24,  25,  false, 0.0f },
        { 29,  26,  false, 0.0f }, { 26,  27,  false, 0.0f }, { 27,  28,  false, 0.0f }, { 28,  29,  false, 0.0f },
        { 33,  30,  false, 0.0f }, { 30,  31,  false, 0.0f }, { 31,  32,  false, 0.0f }, { 32,  33,  false, 0.0f },
        { 38,  34,  false, 0.0f }, { 34,  35,  false, 0.0f }, { 35,  36,  false, 0.0f }, { 36,  37,  false, 0.0f }, { 37,  38,  false, 0.0f },
        { 43,  39,  false, 0.0f }, { 39,  40,  false, 0.0f }, { 40,  41,  false, 0.0f }, { 41,  42,  false, 0.0f }, { 42,  43,  false, 0.0f },
        { 47,  44,  false, 0.0f }, { 44,  45,  false, 0.0f }, { 45,  46,  false, 0.0f }, { 46,  47,  false, 0.0f },
        { 52,  48,  false, 0.0f }, { 48,  49,  false, 0.0f }, { 49,  50,  false, 0.0f }, { 50,  51,  false, 0.0f }, { 51,  52,  false, 0.0f },
        { 56,  53,  false, 0.0f }, { 53,  54,  false, 0.0f }, { 54,  55,  false, 0.0f }, { 55,  56,  false, 0.0f },
        { 60,  57,  false, 0.0f }, { 57,  58,  false, 0.0f }, { 58,  59,  false, 0.0f }, { 59,  60,  false, 0.0f },
        { 64,  61,  false, 0.0f }, { 61,  62,  false, 0.0f }, { 62,  63,  false, 0.0f }, { 63,  64,  false, 0.0f },
    };

    EDGES_T     expected_edges_3            = {
        { 25,  0,   false, 0.0f }, { 0,   1,   false, 0.0f }, { 1,   2,   false, 0.0f }, { 2,   3,   false, 0.0f }, { 3,   4,   false, 0.0f },
        { 4,   5,   false, 0.0f }, { 5,   6,   false, 0.0f }, { 6,   7,   false, 0.0f }, { 7,   8,   false, 0.0f }, { 8,   9,   false, 0.0f },
        { 9,   10,  false, 0.0f }, { 10,  11,  false, 0.0f }, { 11,  12,  false, 0.0f }, { 12,  13,  false, 0.0f }, { 13,  14,  false, 0.0f },
        { 14,  15,  false, 0.0f }, { 15,  16,  false, 0.0f }, { 16,  17,  false, 0.0f }, { 17,  18,  false, 0.0f }, { 18,  19,  false, 0.0f },
        { 19,  20,  false, 0.0f }, { 20,  21,  false, 0.0f }, { 21,  22,  false, 0.0f }, { 22,  23,  false, 0.0f }, { 23,  24,  false, 0.0f },
        { 24,  25,  false, 0.0f },

        { 45,  26,  false, 0.0f }, { 26,  27,  false, 0.0f }, { 27,  28,  false, 0.0f }, { 28,  29,  false, 0.0f }, { 29,  30,  false, 0.0f },
        { 30,  31,  false, 0.0f }, { 31,  32,  false, 0.0f }, { 32,  33,  false, 0.0f }, { 33,  34,  false, 0.0f }, { 34,  35,  false, 0.0f },
        { 35,  36,  false, 0.0f }, { 36,  37,  false, 0.0f }, { 37,  38,  false, 0.0f }, { 38,  39,  false, 0.0f }, { 39,  40,  false, 0.0f },
        { 40,  41,  false, 0.0f }, { 41,  42,  false, 0.0f }, { 42,  43,  false, 0.0f }, { 43,  44,  false, 0.0f }, { 44,  45,  false, 0.0f },

        { 65,  46,  false, 0.0f }, { 46,  47,  false, 0.0f }, { 47,  48,  false, 0.0f }, { 48,  49,  false, 0.0f }, { 49,  50,  false, 0.0f },
        { 50,  51,  false, 0.0f }, { 51,  52,  false, 0.0f }, { 52,  53,  false, 0.0f }, { 53,  54,  false, 0.0f }, { 54,  55,  false, 0.0f },
        { 55,  56,  false, 0.0f }, { 56,  57,  false, 0.0f }, { 57,  58,  false, 0.0f }, { 58,  59,  false, 0.0f }, { 59,  60,  false, 0.0f },
        { 60,  61,  false, 0.0f }, { 61,  62,  false, 0.0f }, { 62,  63,  false, 0.0f }, { 63,  64,  false, 0.0f }, { 64,  65,  false, 0.0f },

        { 75,  66,  false, 0.0f }, { 66,  67,  false, 0.0f }, { 67,  68,  false, 0.0f }, { 68,  69,  false, 0.0f }, { 69,  70,  false, 0.0f },
        { 70,  71,  false, 0.0f }, { 71,  72,  false, 0.0f }, { 72,  73,  false, 0.0f }, { 73,  74,  false, 0.0f }, { 74,  75,  false, 0.0f },

        { 101, 76,  false, 0.0f }, { 76,  77,  false, 0.0f }, { 77,  78,  false, 0.0f }, { 78,  79,  false, 0.0f }, { 79,  80,  false, 0.0f },
        { 80,  81,  false, 0.0f }, { 81,  82,  false, 0.0f }, { 82,  83,  false, 0.0f }, { 83,  84,  false, 0.0f }, { 84,  85,  false, 0.0f },
        { 85,  86,  false, 0.0f }, { 86,  87,  false, 0.0f }, { 87,  88,  false, 0.0f }, { 88,  89,  false, 0.0f }, { 89,  90,  false, 0.0f },
        { 90,  91,  false, 0.0f }, { 91,  92,  false, 0.0f }, { 92,  93,  false, 0.0f }, { 93,  94,  false, 0.0f }, { 94,  95,  false, 0.0f },
        { 95,  96,  false, 0.0f }, { 96,  97,  false, 0.0f }, { 97,  98,  false, 0.0f }, { 98,  99,  false, 0.0f }, { 99,  100,  false, 0.0f },
        { 100, 101, false, 0.0f },
    };

    NUMBERS_T   expected_polygons_begin_1   = {
        0,   4,   8,  12,  16,  20,  24 };

    NUMBERS_T   expected_polygons_end_1     = {
        4,   8,  12,  16,  20,  24,  28 };

    NUMBERS_T   expected_polygons_begin_2   = {
        0,   4,   8,  12,  17,  21,  26,  30,  34,  39,  44,  48,  53,  57,  61 };

    NUMBERS_T   expected_polygons_end_2     = {
        4,   8,  12,  17,  21,  26,  30,  34,  39,  44,  48,  53,  57,  61,  65 };

    NUMBERS_T   expected_polygons_begin_3   = {
        0,  26, 46, 66, 76 };

    NUMBERS_T   expected_polygons_end_3     = {
        26, 46, 66, 76, 102 };

    if (!test(filename_1, expected_vertices_1, expected_edges_1, expected_polygons_begin_1, expected_polygons_end_1))
    {
        fprintf(stderr, "%s:%d:%s: Test failed\n", __FILE__, __LINE__, __FUNCTION__);
        return -1;
    }

    if (!test(filename_2, expected_vertices_2, expected_edges_2, expected_polygons_begin_2, expected_polygons_end_2))
    {
        fprintf(stderr, "%s:%d:%s: Test failed\n", __FILE__, __LINE__, __FUNCTION__);
        return -1;
    }

    if (!test(filename_3, expected_vertices_3, expected_edges_3, expected_polygons_begin_3, expected_polygons_end_3))
    {
        fprintf(stderr, "%s:%d:%s: Test failed\n", __FILE__, __LINE__, __FUNCTION__);
        return -1;
    }

    return 0;
}
