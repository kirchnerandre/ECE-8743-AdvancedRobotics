
#include <cmath>
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

        int32_t final_height    = static_cast<float>(initial_y_max - initial_y_min) / (initial_x_max - initial_x_min) * final_width + 0.5f;

        int32_t final_y_min     = static_cast<float>(final_height) *          final_margin  + 0.5f;
        int32_t final_y_max     = static_cast<float>(final_height) * ( 1.0f - final_margin) + 0.5f;

        Begin.X = final_x_min + static_cast<float>((Begin.X - initial_x_min) * (final_x_max - final_x_min)) / (initial_x_max - initial_x_min) + 0.5f;
        Begin.Y = final_y_min + static_cast<float>((Begin.Y - initial_y_min) * (final_y_max - final_y_min)) / (initial_y_max - initial_y_min) + 0.5f;

        End.Y   = final_y_min + static_cast<float>((End.Y   - initial_y_min) * (final_y_max - final_y_min)) / (initial_y_max - initial_y_min) + 0.5f;
        End.X   = final_x_min + static_cast<float>((End.X   - initial_x_min) * (final_x_max - final_x_min)) / (initial_x_max - initial_x_min) + 0.5f;

        for (IEDGE_T i = Edges.begin(); i != Edges.end(); i++)
        {
            i->VertexA.X = final_x_min + static_cast<float>((i->VertexA.X - initial_x_min) * (final_x_max - final_x_min)) / (initial_x_max - initial_x_min) + 0.5f;
            i->VertexB.X = final_x_min + static_cast<float>((i->VertexB.X - initial_x_min) * (final_x_max - final_x_min)) / (initial_x_max - initial_x_min) + 0.5f;

            i->VertexA.Y = final_y_min + static_cast<float>((i->VertexA.Y - initial_y_min) * (final_y_max - final_y_min)) / (initial_y_max - initial_y_min) + 0.5f;
            i->VertexB.Y = final_y_min + static_cast<float>((i->VertexB.Y - initial_y_min) * (final_y_max - final_y_min)) / (initial_y_max - initial_y_min) + 0.5f;
        }

        Borders = { 0, final_width, 0, final_height };
    }


    void draw_begin(uint8_t* Canvas, BORDERS_T& Borders, VERTEX_T& Begin)
    {
        int32_t channels    = 3;
        int32_t radius      = 5;

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
        int32_t radius      = 5;

        for (int32_t y = End.Y - radius; y <= End.Y + radius; y++)
        {
            for (int32_t x = End.X - radius; x <= End.X + radius; x++)
            {
                if ((Borders.XMin <= x) && (x < Borders.XMax)
                 && (Borders.YMin <= y) && (y < Borders.YMax))
                {
                    Canvas[channels * (y * (Borders.XMax - Borders.XMin) + x) + 2] = 255;
                }
            }
        }
    }


    void draw_edge_vertical_pure(uint8_t* Canvas, BORDERS_T& Borders, EDGE_T& Edge)
    {
        int32_t channels    = 3;
        int32_t radius      = 2;
        int32_t delta_y     = Edge.VertexB.Y >= Edge.VertexA.Y ? 1 : -1;

        for (int32_t y = Edge.VertexA.Y; y <= Edge.VertexB.Y; y += delta_y)
        {
            for (int32_t x = Edge.VertexA.X - radius; x <= Edge.VertexA.X - radius; x++)
            {
                if ((Borders.XMin <= x) && (x < Borders.XMax)
                 && (Borders.YMin <= y) && (y < Borders.YMax))
                {
                    if (Edge.Status)
                    {
                        Canvas[channels * (y * (Borders.XMax - Borders.XMin) + x) + 0] = 255;
                    }
                    else
                    {
                        Canvas[channels * (y * (Borders.XMax - Borders.XMin) + x) + 0] = 255;
                        Canvas[channels * (y * (Borders.XMax - Borders.XMin) + x) + 1] = 255;
                        Canvas[channels * (y * (Borders.XMax - Borders.XMin) + x) + 2] = 255;
                    }
                }
            }
        }
    }


    void draw_edge_vertical_more(uint8_t* Canvas, BORDERS_T& Borders, EDGE_T& Edge)
    {
        int32_t channels    = 3;
        float   radius      = 2.0f;
        float   x           = Edge.VertexA.X;
        float   y           = Edge.VertexA.Y;

        float   step_y      = Edge.VertexB.Y >= Edge.VertexA.Y ? 1.0f : -1.0f;

        float   step_x      = static_cast<float>(Edge.VertexB.X - Edge.VertexA.X)
                                              / (Edge.VertexB.Y - Edge.VertexA.Y);

        while (y <= static_cast<float>(Edge.VertexB.Y))
        {
            for (float delta = - radius; delta <= + radius; delta += 1.0f)
            {
                int32_t coordinate_x = static_cast<int32_t>(x + 0.5f + delta);
                int32_t coordinate_y = static_cast<int32_t>(y + 0.5f);

                if ((Borders.XMin <= coordinate_x) && (coordinate_x < Borders.XMax)
                 && (Borders.YMin <= coordinate_y) && (coordinate_y < Borders.YMax))
                {
                    if (Edge.Status)
                    {
                        Canvas[channels * (coordinate_y * (Borders.XMax - Borders.XMin) + coordinate_x) + 0] = 255;
                    }
                    else
                    {
                        Canvas[channels * (coordinate_y * (Borders.XMax - Borders.XMin) + coordinate_x) + 0] = 255;
                        Canvas[channels * (coordinate_y * (Borders.XMax - Borders.XMin) + coordinate_x) + 1] = 255;
                        Canvas[channels * (coordinate_y * (Borders.XMax - Borders.XMin) + coordinate_x) + 2] = 255;
                    }
                }
            }

            x += step_x;
            y += step_y;
        }
    }


    void draw_edge_horizontal_more(uint8_t* Canvas, BORDERS_T& Borders, EDGE_T& Edge)
    {
        int32_t channels    = 3;
        float   radius      = 2.0f;
        float   x_initial   = 0.0f;
        float   y_initial   = 0.0f;
        float   x_final     = 0.0f;
        float   y_final     = 0.0f;
        float   x_step      = 1.0f;
        float   y_step      = 0.0f;

        if (Edge.VertexA.X < Edge.VertexB.X)
        {
            x_initial   = Edge.VertexA.X;
            x_final     = Edge.VertexB.X;
            y_initial   = Edge.VertexA.Y;
            y_final     = Edge.VertexB.Y;
        }
        else
        {
            x_initial   = Edge.VertexB.X;
            x_final     = Edge.VertexA.X;
            y_initial   = Edge.VertexB.Y;
            y_final     = Edge.VertexA.Y;
        }

        float   x = x_initial;
        float   y = y_initial;

        y_step = (y_final - y_initial) / std::abs(x_final - x_initial);

        while (x <= x_final)
        {
            int32_t coordinate_x = static_cast<int32_t>(x + 0.5f);

            for (float delta = - radius; delta <= + radius; delta += 1.0f)
            {
                int32_t coordinate_y = static_cast<int32_t>(y + 0.5f + delta);

                if ((Borders.XMin <= coordinate_x) && (coordinate_x < Borders.XMax)
                 && (Borders.YMin <= coordinate_y) && (coordinate_y < Borders.YMax))
                {
                    if (Edge.Status)
                    {
                        Canvas[channels * (coordinate_y * (Borders.XMax - Borders.XMin) + coordinate_x) + 0] = 255;
                    }
                    else
                    {
                        Canvas[channels * (coordinate_y * (Borders.XMax - Borders.XMin) + coordinate_x) + 0] = 255;
                        Canvas[channels * (coordinate_y * (Borders.XMax - Borders.XMin) + coordinate_x) + 1] = 255;
                        Canvas[channels * (coordinate_y * (Borders.XMax - Borders.XMin) + coordinate_x) + 2] = 255;
                    }
                }
            }

            x += x_step;
            y += y_step;
        }
    }


    void draw_edge(uint8_t* Canvas, BORDERS_T& Borders, EDGE_T& Edge)
    {
printf("(%d, %d) (%d, %d)\n", Edge.VertexA.X, Edge.VertexA.Y, Edge.VertexB.X, Edge.VertexB.Y);
        int32_t delta_x = Edge.VertexA.X - Edge.VertexB.X >= 0 ? Edge.VertexA.X - Edge.VertexB.X : Edge.VertexB.X - Edge.VertexA.X;
        int32_t delta_y = Edge.VertexA.Y - Edge.VertexB.Y >= 0 ? Edge.VertexA.Y - Edge.VertexB.Y : Edge.VertexB.Y - Edge.VertexA.Y;
printf("%d %d\n", delta_x, delta_y);
        if (delta_x == 0)
        {
printf("a\n");
            draw_edge_vertical_pure(Canvas, Borders, Edge);
        }
        else if (delta_y >= delta_x)
        {
printf("b\n");
            draw_edge_vertical_more(Canvas, Borders, Edge);
        }
        else // if (delta_x < delta_y)
        {
printf("c\n");
            draw_edge_horizontal_more(Canvas, Borders, Edge);
        }
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

    for (IEDGE_T i = Edges.begin(); i != Edges.end(); i++)
    {
        draw_edge(canvas, borders, *i);
    }

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
