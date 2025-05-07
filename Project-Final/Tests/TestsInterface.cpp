
#include "DataTypes.h"
#include "Interface.h"


namespace
{
    bool compare_vertices(VERTICES_T& VerticesA, VERTICES_T& VerticesB)
    {
        if (VerticesA.size() != VerticesB.size())
        {
            fprintf(stderr, "%s:%d:%s: Different vertices size\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        for (size_t i = 0u; i < VerticesA.size(); i++)
        {
            if ((VerticesA[i].X != VerticesB[i].X) || (VerticesA[i].Y != VerticesB[i].Y))
            {
                fprintf(stderr, "%s:%d:%s: Different vertices values\n", __FILE__, __LINE__, __FUNCTION__);
                return false;
            }
        }

        return true;
    }


    bool compare_edges(EDGES_T& EdgesA, EDGES_T& EdgesB)
    {
        if (EdgesA.size() != EdgesB.size())
        {
            fprintf(stderr, "%s:%d:%s: Different edges size\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        for (size_t i = 0u; i < EdgesA.size(); i++)
        {
            if ((EdgesA[i].IndexA != EdgesB[i].IndexA)
             || (EdgesA[i].IndexB != EdgesB[i].IndexB)
             || (EdgesA[i].Status != EdgesB[i].Status))
            {
                fprintf(stderr, "%s:%d:%s: Different edges values\n", __FILE__, __LINE__, __FUNCTION__);
                return false;
            }
        }

        return true;
    }


    bool compare_numbers(NUMBERS_T& NumbersA, NUMBERS_T& NumbersB)
    {
        if (NumbersA.size() != NumbersB.size())
        {
            fprintf(stderr, "%s:%d:%s: Different numbers size\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        for (size_t i = 0u; i < NumbersA.size(); i++)
        {
            if (NumbersA[i] != NumbersB[i])
            {
                fprintf(stderr, "%s:%d:%s: Different numbers values\n", __FILE__, __LINE__, __FUNCTION__);
                return false;
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
            return false;
        }

        if (!compare_vertices(vertices, ExpectedVertices))
        {
            fprintf(stderr, "%s:%d:%s: Invalid vertices\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        if (!compare_edges(edges, ExpectedEdges))
        {
            fprintf(stderr, "%s:%d:%s: Invalid edges\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        if (!compare_numbers(polygons_begin, ExpectedPolygonsBegin))
        {
            fprintf(stderr, "%s:%d:%s: Invalid polygons begin\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        if (!compare_numbers(polygons_end, ExpectedPolygonsEnd))
        {
            fprintf(stderr, "%s:%d:%s: Invalid polygons end\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
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
        {    2.0f,    8.0f }, {    2.0f,   16.0f }, {    4.0f,   16.0f }, {    4.0f,    8.0f },
        {    8.0f,   13.0f }, {    8.0f,   16.0f }, {    9.0f,   16.0f }, {    9.0f,   13.0f },
        {   12.0f,   11.0f }, {   12.0f,   14.0f }, {   15.0f,   14.0f }, {   15.0f,   11.0f },
        {    7.0f,    6.0f }, {    7.0f,   10.0f }, {    9.0f,   10.0f }, {    9.0f,    6.0f },
        {    1.0f,    3.0f }, {    1.0f,    6.0f }, {    4.0f,    6.0f }, {    4.0f,    3.0f },
        {    5.0f,    1.0f }, {    5.0f,    2.0f }, {    9.0f,    2.0f }, {    9.0f,    1.0f },
        {   11.0f,    0.0f }, {   11.0f,    8.0f }, {   13.0f,    8.0f }, {   13.0f,    0.0f } };

    VERTICES_T  expected_vertices_2         = {
        {  600.0f, 1800.0f }, {  650.0f, 1900.0f }, {  750.0f, 1850.0f }, {  850.0f, 1650.0f },
        {  100.0f,  600.0f }, {  180.0f,  700.0f }, {  300.0f,  600.0f }, {  220.0f,  450.0f },
        {  200.0f,  850.0f }, {  250.0f, 1100.0f }, {  350.0f, 1150.0f }, {  350.0f,  750.0f },
        {  200.0f, 1250.0f }, {  100.0f, 1450.0f }, {  250.0f, 1800.0f }, {  400.0f, 1500.0f }, {  350.0f, 1200.0f },
        {  450.0f,   50.0f }, {  350.0f,  200.0f }, {  500.0f,  350.0f }, {  900.0f,   50.0f },
        {  600.0f,  350.0f }, {  550.0f,  500.0f }, {  550.0f,  600.0f }, {  700.0f,  600.0f }, {  850.0f,  500.0f },
        {  450.0f,  850.0f }, {  500.0f, 1100.0f }, {  600.0f, 1200.0f }, {  550.0f,  750.0f },
        {  900.0f,  450.0f }, {  850.0f,  650.0f }, { 1100.0f,  700.0f }, { 1200.0f,  500.0f },
        {  850.0f,  950.0f }, {  800.0f, 1300.0f }, {  900.0f, 1350.0f }, { 1100.0f, 1200.0f }, { 1100.0f,  900.0f },
        { 1250.0f,  600.0f }, { 1300.0f,  700.0f }, { 1400.0f,  650.0f }, { 1500.0f,  500.0f }, { 1350.0f,  350.0f },
        { 1150.0f, 1200.0f }, { 1300.0f, 1250.0f }, { 1400.0f, 1000.0f }, { 1200.0f,  950.0f },
        { 1150.0f, 1550.0f }, { 1050.0f, 1800.0f }, { 1100.0f, 1850.0f }, { 1300.0f, 1750.0f }, { 1250.0f, 1500.0f },
        { 1500.0f,  900.0f }, { 1600.0f,  950.0f }, { 1750.0f,  750.0f }, { 1700.0f,  700.0f },
        { 1400.0f, 1300.0f }, { 1350.0f, 1650.0f }, { 1450.0f, 1700.0f }, { 1650.0f, 1400.0f },
        { 1650.0f, 1050.0f }, { 1700.0f, 1300.0f }, { 1950.0f, 1200.0f }, { 1900.0f,  950.0f } };

    VERTICES_T  expected_vertices_3         = {
        {   10.0f,  140.0f }, {   10.0f,  150.0f }, {   20.0f,  150.0f }, {   20.0f,  170.0f }, {   30.0f,  170.0f },
        {   30.0f,  180.0f }, {   40.0f,  180.0f }, {   40.0f,  190.0f }, {   50.0f,  190.0f }, {   50.0f,  200.0f },
        {   60.0f,  200.0f }, {   60.0f,  190.0f }, {   70.0f,  190.0f }, {   70.0f,  180.0f }, {   80.0f,  180.0f },
        {   80.0f,  150.0f }, {   70.0f,  150.0f }, {   70.0f,  140.0f }, {   60.0f,  140.0f }, {   60.0f,  130.0f },
        {   40.0f,  130.0f }, {   40.0f,  120.0f }, {   30.0f,  120.0f }, {   30.0f,  130.0f }, {   20.0f,  130.0f },
        {   20.0f,  140.0f },

        {   90.0f,  140.0f }, {   90.0f,  150.0f }, {  100.0f,  150.0f }, {  100.0f,  180.0f }, {  120.0f,  180.0f },
        {  120.0f,  190.0f }, {  130.0f,  190.0f }, {  130.0f,  180.0f }, {  150.0f,  180.0f }, {  150.0f,  170.0f },
        {  160.0f,  170.0f }, {  160.0f,  160.0f }, {  150.0f,  160.0f }, {  150.0f,  140.0f }, {  130.0f,  140.0f },
        {  130.0f,  130.0f }, {  110.0f,  130.0f }, {  110.0f,  120.0f }, {  100.0f,  120.0f }, {  100.0f,  140.0f },

        {  160.0f,  100.0f }, {  160.0f,  120.0f }, {  170.0f,  120.0f }, {  170.0f,  130.0f }, {  180.0f,  130.0f },
        {  180.0f,  140.0f }, {  190.0f,  140.0f }, {  190.0f,  150.0f }, {  210.0f,  150.0f }, {  210.0f,  160.0f },
        {  220.0f,  160.0f }, {  220.0f,  140.0f }, {  230.0f,  140.0f }, {  230.0f,  130.0f }, {  220.0f,  130.0f },
        {  220.0f,  100.0f }, {  210.0f,  100.0f }, {  210.0f,   90.0f }, {  180.0f,   90.0f }, {  180.0f,  100.0f },

        {  190.0f,  200.0f }, {  190.0f,  210.0f }, {  200.0f,  210.0f }, {  200.0f,  230.0f }, {  220.0f,  230.0f },
        {  220.0f,  220.0f }, {  230.0f,  220.0f }, {  230.0f,  190.0f }, {  200.0f,  190.0f }, {  200.0f,  200.0f },

        {   70.0f,   50.0f }, {   70.0f,   70.0f }, {   80.0f,   70.0f }, {   80.0f,   80.0f }, {  130.0f,   80.0f },
        {  130.0f,   70.0f }, {  150.0f,   70.0f }, {  150.0f,   60.0f }, {  160.0f,   60.0f }, {  160.0f,   50.0f },
        {  220.0f,   50.0f }, {  220.0f,   40.0f }, {  200.0f,   40.0f }, {  200.0f,   30.0f }, {  220.0f,   30.0f },
        {  220.0f,   20.0f }, {  180.0f,   20.0f }, {  180.0f,   10.0f }, {  170.0f,   10.0f }, {  170.0f,   20.0f },
        {  160.0f,   20.0f }, {  160.0f,   30.0f }, {  150.0f,   30.0f }, {  150.0f,   40.0f }, {   80.0f,   40.0f },
        {   80.0f,   50.0f } };

    EDGES_T     expected_edges_1            = {
        {  3,   0,  false }, {  0,   1,  false }, {  1,   2,  false }, {  2,   3,  false },
        {  7,   4,  false }, {  4,   5,  false }, {  5,   6,  false }, {  6,   7,  false },
        { 11,   8,  false }, {  8,   9,  false }, {  9,  10,  false }, { 10,  11,  false },
        { 15,  12,  false }, { 12,  13,  false }, { 13,  14,  false }, { 14,  15,  false },
        { 19,  16,  false }, { 16,  17,  false }, { 17,  18,  false }, { 18,  19,  false },
        { 23,  20,  false }, { 20,  21,  false }, { 21,  22,  false }, { 22,  23,  false },
        { 27,  24,  false }, { 24,  25,  false }, { 25,  26,  false }, { 26,  27,  false },
    };

    EDGES_T     expected_edges_2            = {
        {  3,   0,  false }, {  0,   1,  false }, {  1,   2,  false }, {  2,   3,  false },
        {  7,   4,  false }, {  4,   5,  false }, {  5,   6,  false }, {  6,   7,  false },
        { 11,   8,  false }, {  8,   9,  false }, {  9,  10,  false }, { 10,  11,  false },
        { 16,  12,  false }, { 12,  13,  false }, { 13,  14,  false }, { 14,  15,  false }, { 15,  16,  false },
        { 20,  17,  false }, { 17,  18,  false }, { 18,  19,  false }, { 19,  20,  false },
        { 25,  21,  false }, { 21,  22,  false }, { 22,  23,  false }, { 23,  24,  false }, { 24,  25,  false },
        { 29,  26,  false }, { 26,  27,  false }, { 27,  28,  false }, { 28,  29,  false },
        { 33,  30,  false }, { 30,  31,  false }, { 31,  32,  false }, { 32,  33,  false },
        { 38,  34,  false }, { 34,  35,  false }, { 35,  36,  false }, { 36,  37,  false }, { 37,  38,  false },
        { 43,  39,  false }, { 39,  40,  false }, { 40,  41,  false }, { 41,  42,  false }, { 42,  43,  false },
        { 47,  44,  false }, { 44,  45,  false }, { 45,  46,  false }, { 46,  47,  false },
        { 52,  48,  false }, { 48,  49,  false }, { 49,  50,  false }, { 50,  51,  false }, { 51,  52,  false },
        { 56,  53,  false }, { 53,  54,  false }, { 54,  55,  false }, { 55,  56,  false },
        { 60,  57,  false }, { 57,  58,  false }, { 58,  59,  false }, { 59,  60,  false },
        { 64,  61,  false }, { 61,  62,  false }, { 62,  63,  false }, { 63,  64,  false },
    };

    EDGES_T     expected_edges_3            = {
        { 25,  0,   false }, { 0,   1,   false }, { 1,   2,   false }, { 2,   3,   false }, { 3,   4,   false },
        { 4,   5,   false }, { 5,   6,   false }, { 6,   7,   false }, { 7,   8,   false }, { 8,   9,   false },
        { 9,   10,  false }, { 10,  11,  false }, { 11,  12,  false }, { 12,  13,  false }, { 13,  14,  false },
        { 14,  15,  false }, { 15,  16,  false }, { 16,  17,  false }, { 17,  18,  false }, { 18,  19,  false },
        { 19,  20,  false }, { 20,  21,  false }, { 21,  22,  false }, { 22,  23,  false }, { 23,  24,  false },
        { 24,  25,  false },

        { 45,  26,  false }, { 26,  27,  false }, { 27,  28,  false }, { 28,  29,  false }, { 29,  30,  false },
        { 30,  31,  false }, { 31,  32,  false }, { 32,  33,  false }, { 33,  34,  false }, { 34,  35,  false },
        { 35,  36,  false }, { 36,  37,  false }, { 37,  38,  false }, { 38,  39,  false }, { 39,  40,  false },
        { 40,  41,  false }, { 41,  42,  false }, { 42,  43,  false }, { 43,  44,  false }, { 44,  45,  false },

        { 65,  46,  false }, { 46,  47,  false }, { 47,  48,  false }, { 48,  49,  false }, { 49,  50,  false },
        { 50,  51,  false }, { 51,  52,  false }, { 52,  53,  false }, { 53,  54,  false }, { 54,  55,  false },
        { 55,  56,  false }, { 56,  57,  false }, { 57,  58,  false }, { 58,  59,  false }, { 59,  60,  false },
        { 60,  61,  false }, { 61,  62,  false }, { 62,  63,  false }, { 63,  64,  false }, { 64,  65,  false },

        { 75,  66,  false }, { 66,  67,  false }, { 67,  68,  false }, { 68,  69,  false }, { 69,  70,  false },
        { 70,  71,  false }, { 71,  72,  false }, { 72,  73,  false }, { 73,  74,  false }, { 74,  75,  false },

        { 101, 76,  false }, { 76,  77,  false }, { 77,  78,  false }, { 78,  79,  false }, { 79,  80,  false },
        { 80,  81,  false }, { 81,  82,  false }, { 82,  83,  false }, { 83,  84,  false }, { 84,  85,  false },
        { 85,  86,  false }, { 86,  87,  false }, { 87,  88,  false }, { 88,  89,  false }, { 89,  90,  false },
        { 90,  91,  false }, { 91,  92,  false }, { 92,  93,  false }, { 93,  94,  false }, { 94,  95,  false },
        { 95,  96,  false }, { 96,  97,  false }, { 97,  98,  false }, { 98,  99,  false }, { 99,  100,  false },
        { 100, 101, false },
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
