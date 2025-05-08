
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
            {  2.0f,  8.0f, false }, {  2.0f, 16.0f, false }, {  4.0f, 16.0f, false }, {  4.0f,  8.0f, false },
            {  8.0f, 13.0f, false }, {  8.0f, 16.0f, false }, {  9.0f, 16.0f, false }, {  9.0f, 13.0f, false },
            { 12.0f, 11.0f, false }, { 12.0f, 14.0f, false }, { 15.0f, 14.0f, false }, { 15.0f, 11.0f, false },
            {  7.0f,  6.0f, false }, {  7.0f, 10.0f, false }, {  9.0f, 10.0f, false }, {  9.0f,  6.0f, false },
            {  1.0f,  3.0f, false }, {  1.0f,  6.0f, false }, {  4.0f,  6.0f, false }, {  4.0f,  3.0f, false },
            {  5.0f,  1.0f, false }, {  5.0f,  2.0f, false }, {  9.0f,  2.0f, false }, {  9.0f,  1.0f, false },
            { 11.0f,  0.0f, false }, { 11.0f,  8.0f, false }, { 13.0f,  8.0f, false }, { 13.0f,  0.0f, false }, };

        EDGES_T edges = {
            { 0,  1,  false, 0.0f }, { 1,  2,  false, 0.0f }, { 2,  3,  false, 0.0f }, { 3,  0,  false, 0.0f },
            { 4,  5,  true,  0.0f }, { 5,  6,  true,  0.0f }, { 6,  7,  true,  0.0f }, { 7,  4,  true,  0.0f },
            { 8,  9,  false, 0.0f }, { 9,  10, false, 0.0f }, { 10, 11, false, 0.0f }, { 11, 8,  false, 0.0f },
            { 12, 13, true,  0.0f }, { 13, 14, true,  0.0f }, { 14, 15, true,  0.0f }, { 15, 12, true,  0.0f },
            { 16, 17, false, 0.0f }, { 17, 18, false, 0.0f }, { 18, 19, false, 0.0f }, { 19, 16, false, 0.0f },
            { 20, 21, true,  0.0f }, { 21, 22, true,  0.0f }, { 22, 23, true,  0.0f }, { 23, 20, true,  0.0f },
            { 24, 25, false, 0.0f }, { 25, 26, false, 0.0f }, { 26, 27, false, 0.0f }, { 27, 24, false, 0.0f } };

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

        EDGES_T edges = {
            { 0,  1,  false, 0.0f }, { 1,  2,  false, 0.0f }, { 2,  3,  false, 0.0f }, { 3,  0,  false, 0.0f },
            { 4,  5,  false, 0.0f }, { 5,  6,  false, 0.0f }, { 6,  7,  false, 0.0f }, { 7,  4,  false, 0.0f },
            { 8,  9,  false, 0.0f }, { 9,  10, false, 0.0f }, { 10, 11, false, 0.0f }, { 11, 8,  false, 0.0f },
            { 12, 13, false, 0.0f }, { 13, 14, false, 0.0f }, { 14, 15, false, 0.0f }, { 15, 16, false, 0.0f }, { 16, 12, false, 0.0f },
            { 17, 18, false, 0.0f }, { 18, 19, false, 0.0f }, { 19, 20, false, 0.0f }, { 20, 17, false, 0.0f },
            { 21, 22, true,  0.0f }, { 22, 23, true,  0.0f }, { 23, 24, true,  0.0f }, { 24, 25, true,  0.0f }, { 25, 21, true,  0.0f },
            { 26, 27, false, 0.0f }, { 27, 28, false, 0.0f }, { 28, 29, false, 0.0f }, { 29, 26, false, 0.0f },
            { 30, 31, false, 0.0f }, { 31, 32, false, 0.0f }, { 32, 33, false, 0.0f }, { 33, 30, false, 0.0f },
            { 34, 35, true,  0.0f }, { 35, 36, true,  0.0f }, { 36, 37, true,  0.0f }, { 37, 38, true,  0.0f }, { 38, 34, true,  0.0f },
            { 39, 40, false, 0.0f }, { 40, 41, false, 0.0f }, { 41, 42, false, 0.0f }, { 42, 43, false, 0.0f }, { 43, 39, false, 0.0f },
            { 44, 45, true,  0.0f }, { 45, 46, true,  0.0f }, { 46, 47, true,  0.0f }, { 47, 44, true,  0.0f },
            { 48, 49, false, 0.0f }, { 49, 50, false, 0.0f }, { 50, 51, false, 0.0f }, { 51, 52, false, 0.0f }, { 52, 48, false, 0.0f },
            { 53, 54, false, 0.0f }, { 54, 55, false, 0.0f }, { 55, 56, false, 0.0f }, { 56, 53, false, 0.0f },
            { 57, 58, true,  0.0f }, { 58, 59, true,  0.0f }, { 59, 60, true,  0.0f }, { 60, 57, true,  0.0f },
            { 61, 62, false, 0.0f }, { 62, 63, false, 0.0f }, { 63, 64, false, 0.0f }, { 64, 61, false, 0.0f } };

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
