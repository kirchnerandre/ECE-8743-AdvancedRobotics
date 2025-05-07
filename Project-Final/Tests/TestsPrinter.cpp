
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
        VERTICES_T vertices = {
            {  2.0f,  8.0f }, {  2.0f, 16.0f }, {  4.0f, 16.0f }, {  4.0f,  8.0f },
            {  8.0f, 13.0f }, {  8.0f, 16.0f }, {  9.0f, 16.0f }, {  9.0f, 13.0f },
            { 12.0f, 11.0f }, { 12.0f, 14.0f }, { 15.0f, 14.0f }, { 15.0f, 11.0f },
            {  7.0f,  6.0f }, {  7.0f, 10.0f }, {  9.0f, 10.0f }, {  9.0f,  6.0f },
            {  1.0f,  3.0f }, {  1.0f,  6.0f }, {  4.0f,  6.0f }, {  4.0f,  3.0f },
            {  5.0f,  1.0f }, {  5.0f,  2.0f }, {  9.0f,  2.0f }, {  9.0f,  1.0f },
            { 11.0f,  0.0f }, { 11.0f,  8.0f }, { 13.0f,  8.0f }, { 13.0f,  0.0f }, };

        EDGES_T edges = {
            { 0,  1,  false }, { 1,  2,  false }, { 2,  3,  false }, { 3,  0,  false },
            { 4,  5,  true  }, { 5,  6,  true  }, { 6,  7,  true  }, { 7,  4,  true  },
            { 8,  9,  false }, { 9,  10, false }, { 10, 11, false }, { 11, 8,  false },
            { 12, 13, true  }, { 13, 14, true  }, { 14, 15, true  }, { 15, 12, true  },
            { 16, 17, false }, { 17, 18, false }, { 18, 19, false }, { 19, 16, false },
            { 20, 21, true  }, { 21, 22, true  }, { 22, 23, true  }, { 23, 20, true  },
            { 24, 25, false }, { 25, 26, false }, { 26, 27, false }, { 27, 24, false } };

        std::string expected    = "../../Data/Testing/expected_map1.ppm";;
        std::string filename    = "./map1.ppm";
        VERTEX_T    begin       = {  0, 16 };
        VERTEX_T    end         = { 16,  0 };

        if (!print(filename, vertices, begin, end, edges))
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
        VERTICES_T vertices = {
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
            { 1650.0f, 1050.0f }, { 1700.0f, 1300.0f }, { 1950.0f, 1200.0f }, { 1900.0f,  950.0f }, };

        EDGES_T edges = {
            { 0,  1,  false }, { 1,  2,  false }, { 2,  3,  false }, { 3,  0,  false },
            { 4,  5,  false }, { 5,  6,  false }, { 6,  7,  false }, { 7,  4,  false },
            { 8,  9,  false }, { 9,  10, false }, { 10, 11, false }, { 11, 8,  false },
            { 12, 13, false }, { 13, 14, false }, { 14, 15, false }, { 15, 16, false }, { 16, 12, false },
            { 17, 18, false }, { 18, 19, false }, { 19, 20, false }, { 20, 17, false },
            { 21, 22, true  }, { 22, 23, true  }, { 23, 24, true  }, { 24, 25, true  }, { 25, 21, true  },
            { 26, 27, false }, { 27, 28, false }, { 28, 29, false }, { 29, 26, false },
            { 30, 31, false }, { 31, 32, false }, { 32, 33, false }, { 33, 30, false },
            { 34, 35, true  }, { 35, 36, true  }, { 36, 37, true  }, { 37, 38, true  }, { 38, 34, true  },
            { 39, 40, false }, { 40, 41, false }, { 41, 42, false }, { 42, 43, false }, { 43, 39, false },
            { 44, 45, true  }, { 45, 46, true  }, { 46, 47, true  }, { 47, 44, true  },
            { 48, 49, false }, { 49, 50, false }, { 50, 51, false }, { 51, 52, false }, { 52, 48, false },
            { 53, 54, false }, { 54, 55, false }, { 55, 56, false }, { 56, 53, false },
            { 57, 58, true  }, { 58, 59, true  }, { 59, 60, true  }, { 60, 57, true  },
            { 61, 62, false }, { 62, 63, false }, { 63, 64, false }, { 64, 61, false } };

        std::string expected    = "../../Data/Testing/expected_map2.ppm";
        std::string filename    = "./map2.ppm";
        VERTEX_T    begin       = {  200,  200 };
        VERTEX_T    end         = { 1800, 1800 };

        if (!print(filename, vertices, begin, end, edges))
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
}
