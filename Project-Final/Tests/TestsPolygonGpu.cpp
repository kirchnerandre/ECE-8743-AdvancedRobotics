
#include <stdio.h>

#include "DataTypes.h"
#include "PolygonGpu.h"


namespace
{
    bool test_1()
    {
        VERTEX_T    vertices[]          = {
            { 400.0f, 200.0f }, { 800.0f, 200.0f }, { 800.0f, 400.0f }, { 400.0f, 400.0f }, { 300.0f, 300.0f }, { 900.0f, 300.0f } };

        EDGE_T      edge                = { 4, 5, false, 0.0f };

        NUMBER_T    polygons_begin[]    = { 0 };

        NUMBER_T    polygons_end[]      = { 4 };

        BOOL_T      statuses[]          = { false };

        BOOL_T      expected[]          = { true };

        if (!test_colision(statuses, polygons_begin, polygons_end, vertices, edge, 6u, 1u))
        {
            fprintf(stderr, "%s:%d:%s: test_colision failed\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        if (statuses[0] != expected[0])
        {
            fprintf(stderr, "%s:%d:%s: Incorrect status\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        return true;
    }


    bool test_2()
    {
        VERTEX_T    vertices[]          = {
            { 100, 200 }, { 200, 200 }, { 200, 400 }, { 100, 400 },
            { 400, 200 }, { 800, 200 }, { 800, 400 }, { 400, 400 },
            { 300, 300 }, { 900, 300 } };

        EDGE_T      edge                = { 8, 9, false, 0.0f };

        NUMBER_T    polygons_begin[]    = { 0, 4 };

        NUMBER_T    polygons_end[]      = { 4, 8 };

        BOOL_T      statuses[]          = { false, false };

        BOOL_T      expected[]          = { false, true };

        if (!test_colision(statuses, polygons_begin, polygons_end, vertices, edge, 10u, 2u))
        {
            fprintf(stderr, "%s:%d:%s: test_colision failed\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        if (statuses[0] != expected[0])
        {
            fprintf(stderr, "%s:%d:%s: Incorrect status\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        for (int32_t i = 0; i < sizeof (statuses) / sizeof(BOOL_T); i++)
        {
            if (statuses[i] != expected[i])
            {
                fprintf(stderr, "%s:%d:%s: Incorrect status\n", __FILE__, __LINE__, __FUNCTION__);
                return false;
            }
        }

        return true;
    }


    bool test_3()
    {
        VERTEX_T    vertices[]      = {
            {  600, 1800, false }, {  650, 1900, false }, {  750, 1850, false }, {  850, 1650, false },
            {  100,  600, false }, {  180,  700, false }, {  300,  600, false }, {  220,  450, false },
            {  200,  850, false }, {  250, 1100, false }, {  350, 1150, false }, {  350,  750, false },
            {  200, 1250, false }, {  100, 1450, false }, {  250, 1800, false }, {  400, 1500, false }, {  350, 1200, false },
            {  450,   50, false }, {  350,  200, false }, {  500,  350, false }, {  900,   50, false },
            {  600,  350, false }, {  550,  500, false }, {  550,  600, false }, {  700,  600, false }, {  850,  500, false },
            {  450,  850, false }, {  500, 1100, false }, {  600, 1200, false }, {  550,  750, false },
            {  900,  450, false }, {  850,  650, false }, { 1100,  700, false }, { 1200,  500, false },
            {  850,  950, false }, {  800, 1300, false }, {  900, 1350, false }, { 1100, 1200, false }, { 1100,  900, false },
            { 1250,  600, false }, { 1300,  700, false }, { 1400,  650, false }, { 1500,  500, false }, { 1350,  350, false },
            { 1150, 1200, false }, { 1300, 1250, false }, { 1400, 1000, false }, { 1200,  950, false },
            { 1150, 1550, false }, { 1050, 1800, false }, { 1100, 1850, false }, { 1300, 1750, false }, { 1250, 1500, false },
            { 1500,  900, false }, { 1600,  950, false }, { 1750,  750, false }, { 1700,  700, false },
            { 1400, 1300, false }, { 1350, 1650, false }, { 1450, 1700, false }, { 1650, 1400, false },
            { 1650, 1050, false }, { 1700, 1300, false }, { 1950, 1200, false }, { 1900,  950, false },

            { 200,  200,  false }, { 1800, 1800, false } };

        EDGE_T      edge                = { 65, 66, false, 0.0f };

        NUMBER_T    polygons_begin[]    = {
            0,  4,  8,  12, 17,
            21, 26, 30, 34, 39,
            44, 48, 53, 57, 61 };

        NUMBER_T    polygons_end[]      = {
            4,  8,  12, 17, 21,
            26, 30, 34, 39, 44,
            48, 53, 57, 61, 65 };

        BOOL_T      statuses[]          = {
            false, false, false, false, false,
            false, false, false, false, false,
            false, false, false, false, false };

        BOOL_T      expected[]          = {
            false, false, false, false, false,
            true,  false, false, true,  false,
            true,  false, false, true,  false };

        if (!test_colision(statuses, polygons_begin, polygons_end, vertices, edge, sizeof(vertices) / sizeof(VERTEX_T), sizeof(polygons_begin) / sizeof(NUMBER_T)))
        {
            fprintf(stderr, "%s:%d:%s: test_colision failed\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        for (int32_t i = 0; i < sizeof (statuses) / sizeof(BOOL_T); i++)
        {
            if (statuses[i] != expected[i])
            {
                fprintf(stderr, "%s:%d:%s: Incorrect status\n", __FILE__, __LINE__, __FUNCTION__);
                return false;
            }
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
