
#include <stdio.h>

#include "Printer.h"


namespace
{
    bool compare(std::string& MapA, std::string& MapB)
    {
        bool    ret_val = true;
        FILE*   file_a  = nullptr;
        FILE*   file_b  = nullptr;

        if (fopen_s(&file_a, MapA.c_str(), "r"))
        {
            fprintf(stderr, "%s:%d:%s: Failed to open file\n", __FILE__, __LINE__, __FUNCTION__);
            ret_val = false;
            goto terminate;
        }

        if (fopen_s(&file_b, MapB.c_str(), "r"))
        {
            fprintf(stderr, "%s:%d:%s: Failed to open file\n", __FILE__, __LINE__, __FUNCTION__);
            ret_val = false;
            goto terminate;
        }

        while (1)
        {
            char symbol_a = getc(file_a);
            char symbol_b = getc(file_b);

            if (symbol_a != symbol_b)
            {
                fprintf(stderr, "%s:%d:%s: Failed symbols\n", __FILE__, __LINE__, __FUNCTION__);
                ret_val = false;
                goto terminate;
            }
            else if (symbol_a == EOF)
            {
                break;
            }
        }

terminate:
        if (file_a)
        {
            fclose(file_a);
        }

        if (file_b)
        {
            fclose(file_b);
        }

        return ret_val;
    }


    bool test_1()
    {
        EDGES_T edges = {
            { {  2,  8 }, {  2, 16 }, false },
            { {  2, 16 }, {  4, 16 }, false },
            { {  4, 16 }, {  4,  8 }, false },
            { {  4,  8 }, {  2,  8 }, false },

            { {  8, 13 }, {  8, 16 }, true },
            { {  8, 16 }, {  9, 16 }, true },
            { {  9, 16 }, {  9, 13 }, true },
            { {  9, 13 }, {  8, 13 }, true },

            { { 12, 11 }, { 12, 14 }, false },
            { { 12, 14 }, { 15, 14 }, false },
            { { 15, 14 }, { 15, 11 }, false },
            { { 15, 11 }, { 12, 11 }, false },

            { {  7,  6 }, {  7, 10 }, true },
            { {  7, 10 }, {  9, 10 }, true },
            { {  9, 10 }, {  9,  6 }, true },
            { {  9,  6 }, {  7,  6 }, true },

            { {  1,  3 }, {  1,  6 }, false },
            { {  1,  6 }, {  4,  6 }, false },
            { {  4,  6 }, {  4,  3 }, false },
            { {  4,  3 }, {  1,  3 }, false },

            { {  5,  1 }, {  5,  2 }, true },
            { {  5,  2 }, {  9,  2 }, true },
            { {  9,  2 }, {  9,  1 }, true },
            { {  9,  1 }, {  5,  1 }, true },

            { { 11,  0 }, { 11,  8 }, false },
            { { 11,  8 }, { 13,  8 }, false },
            { { 13,  8 }, { 13,  0 }, false },
            { { 13,  0 }, { 11,  0 }, false } };

        std::string expected    = "../../Data/Testing/expected_map1.ppm";;
        std::string filename    = "./map1.ppm";
        VERTEX_T    begin       = {  0, 16 };
        VERTEX_T    end         = { 16,  0 };

        if (!print(filename, begin, end, edges))
        {
            fprintf(stderr, "%s:%d:%s: Print map failed\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        if (!compare(expected, filename))
        {
            fprintf(stderr, "%s:%d:%s: Invalid generated picture\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        return true;
    }


    bool test_2()
    {
        EDGES_T edges = {
            { {  200,  200 }, { 1800, 1800 }, true },

            { {  600, 1800 }, {  650, 1900 }, false },
            { {  650, 1900 }, {  750, 1850 }, false },
            { {  750, 1850 }, {  850, 1650 }, false },
            { {  850, 1650 }, {  600, 1800 }, false },

            { {  100,  600 }, {  180,  700 }, false },
            { {  180,  700 }, {  300,  600 }, false },
            { {  300,  600 }, {  220,  450 }, false },
            { {  220,  450 }, {  100,  600 }, false },

            { {  200,  850 }, {  250, 1100 }, false },
            { {  250, 1100 }, {  350, 1150 }, false },
            { {  350, 1150 }, {  350,  750 }, false },
            { {  350,  750 }, {  200,  850 }, false },

            { {  200, 1250 }, {  100, 1450 }, false },
            { {  100, 1450 }, {  250, 1800 }, false },
            { {  250, 1800 }, {  400, 1500 }, false },
            { {  400, 1500 }, {  350, 1200 }, false },
            { {  350, 1200 }, {  200, 1250 }, false },

            { {  450,   50 }, {  350,  200 }, false },
            { {  350,  200 }, {  500,  350 }, false },
            { {  500,  350 }, {  900,   50 }, false },
            { {  900,   50 }, {  450,   50 }, false },

            { {  600,  350 }, {  550,  500 }, true }, // 6
            { {  550,  500 }, {  550,  600 }, true },
            { {  550,  600 }, {  700,  600 }, true },
            { {  700,  600 }, {  850,  500 }, true },
            { {  850,  500 }, {  600,  350 }, true },

            { {  450,  850 }, {  500, 1100 }, false },
            { {  500, 1100 }, {  600, 1200 }, false },
            { {  600, 1200 }, {  550,  750 }, false },
            { {  550,  750 }, {  450,  850 }, false },

            { {  900,  450 }, {  850,  650 }, false },
            { {  850,  650 }, { 1100,  700 }, false },
            { { 1100,  700 }, { 1200,  500 }, false },
            { { 1200,  500 }, {  900,  450 }, false },

            { {  850,  950 }, {  800, 1300 }, true }, // 9
            { {  800, 1300 }, {  900, 1350 }, true },
            { {  900, 1350 }, { 1100, 1200 }, true },
            { { 1100, 1200 }, { 1100,  900 }, true },
            { { 1100,  900 }, {  850,  950 }, true },

            { { 1250,  600 }, { 1300,  700 }, false },
            { { 1300,  700 }, { 1400,  650 }, false },
            { { 1400,  650 }, { 1500,  500 }, false },
            { { 1500,  500 }, { 1350,  350 }, false },
            { { 1350,  350 }, { 1250,  600 }, false },

            { { 1150, 1200 }, { 1300, 1250 }, true }, // 11
            { { 1300, 1250 }, { 1400, 1000 }, true },
            { { 1400, 1000 }, { 1200,  950 }, true },
            { { 1200,  950 }, { 1150, 1200 }, true },

            { { 1150, 1550 }, { 1050, 1800 }, false },
            { { 1050, 1800 }, { 1100, 1850 }, false },
            { { 1100, 1850 }, { 1300, 1750 }, false },
            { { 1300, 1750 }, { 1250, 1500 }, false },
            { { 1250, 1500 }, { 1150, 1550 }, false },

            { { 1500,  900 }, { 1600,  950 }, false },
            { { 1600,  950 }, { 1750,  750 }, false },
            { { 1750,  750 }, { 1700,  700 }, false },
            { { 1700,  700 }, { 1500,  900 }, false },

            { { 1400, 1300 }, { 1350, 1650 }, true }, // 14
            { { 1350, 1650 }, { 1450, 1700 }, true },
            { { 1450, 1700 }, { 1650, 1400 }, true },
            { { 1650, 1400 }, { 1400, 1300 }, true },

            { { 1650, 1050 }, { 1700, 1300 }, false },
            { { 1700, 1300 }, { 1950, 1200 }, false },
            { { 1950, 1200 }, { 1900,  950 }, false },
            { { 1900,  950 }, { 1650, 1050 }, false } };

        std::string expected    = "../../Data/Testing/expected_map2.ppm";
        std::string filename    = "./map2.ppm";
        VERTEX_T    begin       = {  200,  200 };
        VERTEX_T    end         = { 1800, 1800 };

        if (!print(filename, begin, end, edges))
        {
            fprintf(stderr, "%s:%d:%s: Print map failed\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        if (!compare(expected, filename))
        {
            fprintf(stderr, "%s:%d:%s: Invalid generated picture\n", __FILE__, __LINE__, __FUNCTION__);
            return false;
        }

        return true;
    }
}


int main()
{
#if 0
    if (!test_1())
    {
        fprintf(stderr, "%s:%d:%s: Test failed\n", __FILE__, __LINE__, __FUNCTION__);
        return -1;
    }
#endif
    if (!test_2())
    {
        fprintf(stderr, "%s:%d:%s: Test failed\n", __FILE__, __LINE__, __FUNCTION__);
        return -1;
    }
}
