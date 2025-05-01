
#include <stdio.h>
#include <string>

#include "DataTypes.h"


namespace
{
    void get_number(int32_t& Number, FILE* File)
    {
        std::string number;

        while (1)
        {
            char symbol = getc(File);

            if ((symbol == ' ') || (symbol == '\n') || (symbol == EOF))
            {
                if (number.size())
                {
                    Number = std::stoi(number);
                    break;
                }
            }
            else
            {
                number += symbol;
            }
        }
    }


    bool get_vertices(VERTICES_T& Vertices, size_t Numbers, std::string& Filename)
    {
        bool    ret_val = true;
        FILE*   file    = nullptr;
        std::string number;

        if (fopen_s(&file, Filename.c_str(), "r"))
        {
            fprintf(stderr, "%s:%d:%s: Failed to open file\n", __FILE__, __LINE__, __FUNCTION__);
            ret_val = false;
            goto terminate;
        }

        Vertices.reserve(Numbers / 2);

        for (size_t i = 0u; i < Numbers / 2; i++)
        {
            VERTEX_T vertex{};

            get_number(vertex.X, file);
            get_number(vertex.Y, file);

            Vertices.push_back(vertex);
        }

    terminate:
        if (file)
        {
            fclose(file);
        }

        return ret_val;
    }


    bool count_lines(size_t& Lines, std::string& Filename)
    {
        bool    ret_val = true;
        FILE*   file    = nullptr;

        Lines = 0u;

        if (fopen_s(&file, Filename.c_str(), "r"))
        {
            fprintf(stderr, "%s:%d:%s: Failed to open file\n", __FILE__, __LINE__, __FUNCTION__);
            ret_val = false;
            goto terminate;
        }

        while (1)
        {
            char symbols = getc(file);

            if (symbols == '\n')
            {
                Lines++;
            }
            else if (symbols == EOF)
            {
                break;
            }
        }

    terminate:
        if (file)
        {
            fclose(file);
        }

        return ret_val;
    }


    bool count_numbers(size_t& Numbers, std::string& Filename)
    {
        bool        ret_val = true;
        FILE*       file    = nullptr;
        std::string number;

        Numbers = 0u;

        if (fopen_s(&file, Filename.c_str(), "r"))
        {
            fprintf(stderr, "%s:%d:%s: Failed to open file\n", __FILE__, __LINE__, __FUNCTION__);
            ret_val = false;
            goto terminate;
        }

        while (1)
        {
            char symbol = getc(file);

            if ((symbol == ' ') || (symbol == '\n') || (symbol == EOF))
            {
                if (number.size())
                {
                    Numbers++;
                    number = "";
                }

                if (symbol == EOF)
                {
                    break;
                }
            }
            else
            {
                number += symbol;
            }
        }

    terminate:
        if (file)
        {
            fclose(file);
        }

        return ret_val;
    }
}


bool read_map(VERTICES_T& Vertices, size_t& Polygons, std::string Filename)
{
    if (!count_lines(Polygons, Filename))
    {
        fprintf(stderr, "%s:%d:%s: Failed to get number of lines\n", __FILE__, __LINE__, __FUNCTION__);
        return false;
    }

    size_t numbers = 0u;

    if (!count_numbers(numbers, Filename))
    {
        fprintf(stderr, "%s:%d:%s: Failed to get number of numbers\n", __FILE__, __LINE__, __FUNCTION__);
        return false;
    }

    if (!get_vertices(Vertices, numbers, Filename))
    {
        fprintf(stderr, "%s:%d:%s: Failed to get vertices\n", __FILE__, __LINE__, __FUNCTION__);
        return false;
    }

    return true;
}
