
#include <stdio.h>

#include "Printer.h"


namespace
{
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
    BORDERS_T&      Borders,
    VERTEX_T&       Begin,
    VERTEX_T&       End,
    EDGES_T&        Edges)
{
    constexpr uint32_t  channels    = 3u;
    int32_t             size        = (Borders.XMax - Borders.XMin)
                                    * (Borders.YMax - Borders.YMin) * channels;
    uint8_t*            canvas      = new uint8_t[size]{};
    bool                ret_val     = true;

canvas[0]   = 255;
canvas[1]   = 255;
canvas[9]   = 255;
canvas[100] = 255;
print_canvas(canvas, Borders);

//   draw_begin(canvas, Borders, Begin);
//print_canvas(canvas, Borders);
//  draw_end  (canvas, Borders, End);
//print_canvas(canvas, Borders);

    for (IEDGE_T i = Edges.begin(); i != Edges.end(); i++)
    {
        draw_edge(canvas, Borders, *i);
    }

    FILE* file = nullptr;

    if (fopen_s(&file, Filename.c_str(), "w"))
    {
        fprintf(stderr, "%s:%d:%s: Failed to create file\n", __FILE__, __LINE__, __FUNCTION__);
        ret_val = false;
        goto terminate;
    }

    fprintf(file, "P3 ");
    fprintf(file, "%d  ", Borders.XMax - Borders.XMin);
    fprintf(file, "%d\n", Borders.YMax - Borders.YMin);
    fprintf(file, "255\n");

    int32_t height = Borders.YMax - Borders.YMin;
    int32_t width  = Borders.XMax - Borders.XMin;

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
