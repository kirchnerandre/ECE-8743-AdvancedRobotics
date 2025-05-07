
#include <stdio.h>
#include <string>

#include "DataTypes.h"


namespace
{
    void get_edges(EDGES_T& Edges, NUMBER_T PolygonBegin, NUMBER_T PolygonEnd)
    {
        for (NUMBER_T i = PolygonBegin; i < PolygonEnd; i++)
        {
            EDGE_T edge{};

            if (i == PolygonBegin)
            {
                edge.IndexA = PolygonEnd - 1;
                edge.IndexB = i;
            }
            else
            {
                edge.IndexA = i - 1;
                edge.IndexB = i;
            }

            Edges.push_back(edge);
        }
    }


    void get_number(int32_t& Number, size_t& Offset, std::string& Line)
    {
        std::string number;

        while (1)
        {
            if (Offset >= Line.size())
            {
                break;
            }
            else if ('0' <= (Line.at(Offset)) && (Line.at(Offset) <= '9'))
            {
                number += Line.at(Offset);
            }
            else if (number.size())
            {
                break;
            }

            Offset++;
        }

        Number = std::stoi(number);
    }


    void get_number(float& Number, size_t& Offset, std::string& Line)
    {
        std::string number;

        while (1)
        {
            if (Offset >= Line.size())
            {
                break;
            }
            else if ('0' <= (Line.at(Offset)) && (Line.at(Offset) <= '9'))
            {
                number += Line.at(Offset);
            }
            else if (number.size())
            {
                break;
            }

            Offset++;
        }

        Number = std::stof(number);
    }


    bool get_line(std::string& Line, FILE* File)
    {
        while (1)
        {
            char symbol = getc(File);

            if (symbol == EOF)
            {
                return false;
            }
            else if (symbol == '\n')
            {
                return true;
            }
            else
            {
                Line += symbol;
            }
        }
    }
}


bool read_map(VERTICES_T& Vertices, EDGES_T& Edges, NUMBERS_T& PolygonsBegin, NUMBERS_T& PolygonsEnd, std::string Filename)
{
    bool        ret_val = true;
    FILE*       file    = nullptr;

    if (fopen_s(&file, Filename.c_str(), "r"))
    {
        fprintf(stderr, "%s:%d:%s: Failed to open file\n", __FILE__, __LINE__, __FUNCTION__);
        ret_val = false;
        goto terminate;
    }

    while (1)
    {
        std::string line;

        if (!get_line(line, file))
        {
            break;
        }

        NUMBER_T    size    = 0;
        NUMBER_T    zero    = 0;
        size_t      offset  = 0u;

        get_number(size, offset, line);
        get_number(zero, offset, line);

        PolygonsEnd  .push_back(Vertices.size() + size);
        PolygonsBegin.push_back(Vertices.size());

        get_edges(Edges, PolygonsBegin.back(), PolygonsEnd.back());

        for (NUMBER_T i = 0; i < size; i++)
        {
            VERTEX_T vertex{};

            get_number(vertex.X, offset, line);
            get_number(vertex.Y, offset, line);

            Vertices.push_back(vertex);
        }
    }

terminate:
    if (file)
    {
        fclose(file);
    }

    return true;
}
