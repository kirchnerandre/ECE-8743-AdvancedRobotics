
#include <stdio.h>

#include "Printer.h"


namespace
{
    struct BORDERS_T
    {
        int32_t XMin;
        int32_t XMax;
        int32_t YMin;
        int32_t YMax;
    };


    void scale(BORDERS_T& Borders, VERTEX_T& Begin, VERTEX_T& End, EDGES_T& Edges)
    {
        constexpr float     final_margin    = 0.05f;
        constexpr int32_t   final_width     = 1000;
        constexpr int32_t   final_x_min     = 50;
        constexpr int32_t   final_x_max     = 950;
        int32_t             initial_x_min   = Begin.X;
        int32_t             initial_x_max   = Begin.X;
        int32_t             initial_y_min   = Begin.Y;
        int32_t             initial_y_max   = Begin.Y;

        initial_x_min = End.X < initial_x_min ? End.X : initial_x_min;
        initial_x_max = End.X > initial_x_max ? End.X : initial_x_max;

        initial_y_min = End.Y < initial_y_min ? End.Y : initial_y_min;
        initial_y_max = End.Y > initial_y_max ? End.Y : initial_y_max;

        for (IEDGE_T i = Edges.begin(); i != Edges.end(); i++)
        {
            initial_x_min = i->VertexA.X < initial_x_min ? i->VertexA.X : initial_x_min;
            initial_x_max = i->VertexA.X > initial_x_max ? i->VertexA.X : initial_x_max;

            initial_x_min = i->VertexB.X < initial_x_min ? i->VertexB.X : initial_x_min;
            initial_x_max = i->VertexB.X > initial_x_max ? i->VertexB.X : initial_x_max;

            initial_y_min = i->VertexA.Y < initial_y_min ? i->VertexA.Y : initial_y_min;
            initial_y_max = i->VertexA.Y > initial_y_max ? i->VertexA.Y : initial_y_max;

            initial_y_min = i->VertexB.Y < initial_y_min ? i->VertexB.Y : initial_y_min;
            initial_y_max = i->VertexB.Y > initial_y_max ? i->VertexB.Y : initial_y_max;
        }

        int32_t final_height    = (initial_y_max - initial_y_min) / (initial_x_max - initial_x_min) * final_width;

        int32_t final_y_min     = final_height *          final_margin;
        int32_t final_y_max     = final_height * ( 1.0f - final_margin);

        for (IEDGE_T i = Edges.begin(); i != Edges.end(); i++)
        {
            i->VertexA.X = final_x_min + static_cast<float>((i->VertexA.X - initial_x_min) * (final_x_max - final_x_min)) / (initial_x_max - initial_x_min) + 0.5f;
            i->VertexB.X = final_x_min + static_cast<float>((i->VertexB.X - initial_x_min) * (final_x_max - final_x_min)) / (initial_x_max - initial_x_min) + 0.5f;

            i->VertexA.Y = final_y_min + static_cast<float>((i->VertexA.Y - initial_y_min) * (final_y_max - final_y_min)) / (initial_y_max - initial_y_min) + 0.5f;
            i->VertexB.Y = final_y_min + static_cast<float>((i->VertexB.Y - initial_y_min) * (final_y_max - final_y_min)) / (initial_y_max - initial_y_min) + 0.5f;
        }

        Borders = { 0, final_width, 0, final_height };
    }


    void draw()
    {

    }


    void print_canvas(uint8_t* Canvas, BORDERS_T& Borders)
    {
        int32_t channels    = 3;
        int32_t width       = Borders.XMax - Borders.XMin;
        int32_t height      = Borders.YMax - Borders.YMin;

        for (int32_t y = 0; y < height; y++)
        {
            for (int32_t x = 0; x < width; x++)
            {
                printf("(%3d %3d %3d) ", Canvas[channels * (y * width + x) + 0],
                                         Canvas[channels * (y * width + x) + 1],
                                         Canvas[channels * (y * width + x) + 2]);
            }

            printf("\n");
        }

        printf("\n");
    }


    void draw_begin(uint8_t* Canvas, BORDERS_T& Borders, VERTEX_T& Begin)
    {
        int32_t channels    = 3;
        int32_t radius      = 2;

        for (int32_t y = Begin.Y - radius; y <= Begin.Y + radius; y++)
        {
            for (int32_t x = Begin.X - radius; x <= Begin.X + radius; x++)
            {
                if ((Borders.XMin <= x) && (x < Borders.XMax)
                 && (Borders.YMin <= y) && (y < Borders.YMax))
                {
                    Canvas[channels * (y * (Borders.XMax - Borders.XMin) + x) + 1] = 255;
                }
            }
        }
    }


    void draw_end(uint8_t* Canvas, BORDERS_T& Borders, VERTEX_T& End)
    {
        int32_t channels    = 3;
        int32_t radius      = 2;

        for (int32_t y = End.Y - radius; y <= End.Y + radius; y++)
        {
            for (int32_t x = End.X - radius; x <= End.X + radius; x++)
            {
                if ((Borders.XMin <= x) && (x < Borders.XMax)
                 && (Borders.YMin <= y) && (y < Borders.YMax))
                {
                    Canvas[channels * (y * (Borders.XMax - Borders.XMin) + x) + 0] = 255;
                }
            }
        }
    }


    void draw_edge(uint8_t* Canvas, BORDERS_T& Borders, EDGE_T& Edge)
    {

    }
}


bool print(
    std::string&    Filename,
    VERTEX_T&       Begin,
    VERTEX_T&       End,
    EDGES_T&        Edges)
{
    constexpr uint32_t  channels    = 3u;

    bool                ret_val     = true;
    uint8_t*            canvas      = nullptr;
    int32_t             width       = 0;
    int32_t             height      = 0;
    BORDERS_T           borders{};

    scale(borders, Begin, End, Edges);

    height = borders.YMax - borders.YMin;
    width  = borders.XMax - borders.XMin;

    canvas = new uint8_t[channels * width * height]{};

    draw_begin(canvas, borders, Begin);
    draw_end  (canvas, borders, End);

    FILE* file = nullptr;

    if (fopen_s(&file, Filename.c_str(), "w"))
    {
        fprintf(stderr, "%s:%d:%s: Failed to create file\n", __FILE__, __LINE__, __FUNCTION__);
        ret_val = false;
        goto terminate;
    }

    fprintf(file, "P3 ");
    fprintf(file, "%d  ", borders.XMax - borders.XMin);
    fprintf(file, "%d\n", borders.YMax - borders.YMin);
    fprintf(file, "255\n");

    for (int32_t y = 0; y < height; y++)
    {
        for (int32_t x = 0; y < width; x++)
        {
            fprintf(file, "%4d", canvas[channels * (y * width + x) + 0]);
            fprintf(file, "%4d", canvas[channels * (y * width + x) + 1]);
            fprintf(file, "%4d", canvas[channels * (y * width + x) + 2]);
        }
    }

terminate:
    if (canvas)
    {
        delete[] canvas;
    }

    if (file)
    {
        fclose(file);
    }

    return ret_val;
}
