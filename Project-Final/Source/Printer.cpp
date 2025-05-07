
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


    void scale(BORDERS_T& Borders, VERTICES_T& Vertices, VERTEX_T& Begin, VERTEX_T& End)
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

        for (size_t i = 0u; i < Vertices.size(); i++)
        {
            initial_x_min = Vertices[i].X < initial_x_min ? Vertices[i].X : initial_x_min;
            initial_x_max = Vertices[i].X > initial_x_max ? Vertices[i].X : initial_x_max;

            initial_y_min = Vertices[i].Y < initial_y_min ? Vertices[i].Y : initial_y_min;
            initial_y_max = Vertices[i].Y > initial_y_max ? Vertices[i].Y : initial_y_max;
        }

        int32_t final_height    = static_cast<float>(initial_y_max - initial_y_min) / (initial_x_max - initial_x_min) * final_width;

        int32_t final_y_min     = static_cast<float>(final_height) *          final_margin;
        int32_t final_y_max     = static_cast<float>(final_height) * ( 1.0f - final_margin);

        Begin.X = final_x_min + static_cast<float>((Begin.X - initial_x_min) * (final_x_max - final_x_min)) / (initial_x_max - initial_x_min);
        Begin.Y = final_y_min + static_cast<float>((Begin.Y - initial_y_min) * (final_y_max - final_y_min)) / (initial_y_max - initial_y_min);

        End.Y   = final_y_min + static_cast<float>((End.Y   - initial_y_min) * (final_y_max - final_y_min)) / (initial_y_max - initial_y_min);
        End.X   = final_x_min + static_cast<float>((End.X   - initial_x_min) * (final_x_max - final_x_min)) / (initial_x_max - initial_x_min);

        for (size_t i = 0u; i < Vertices.size(); i++)
        {
            Vertices[i].X = final_x_min + static_cast<float>((Vertices[i].X - initial_x_min) * (final_x_max - final_x_min)) / (initial_x_max - initial_x_min);
            Vertices[i].Y = final_y_min + static_cast<float>((Vertices[i].Y - initial_y_min) * (final_y_max - final_y_min)) / (initial_y_max - initial_y_min);
        }

        Borders = { 0, final_width, 0, final_height };
    }


    void draw_begin(uint8_t* Canvas, BORDERS_T& Borders, VERTEX_T& Begin)
    {
        int32_t channels    = 3;
        int32_t radius      = 8;

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
        int32_t radius      = 8;

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


    void draw_edge_vertical_pure(uint8_t* Canvas, BORDERS_T& Borders, VERTICES_T& Vertices, EDGE_T& Edge)
    {
        int32_t channels    = 3;
        float   radius      = 2.0f;
        float   y_initial   = 0.0f;
        float   y_final     = 0.0f;
        float   y_step      = 1.0f;
        float   x           = Vertices[Edge.IndexB].X;

        if (Vertices[Edge.IndexA].Y < Vertices[Edge.IndexB].Y)
        {
            y_initial   = Vertices[Edge.IndexA].Y;
            y_final     = Vertices[Edge.IndexB].Y;
        }
        else
        {
            y_initial   = Vertices[Edge.IndexB].Y;
            y_final     = Vertices[Edge.IndexA].Y;
        }

        for (float y = y_initial; y <= y_final; y += y_step)
        {
            int32_t coordinate_y = static_cast<int32_t>(y + 0.5f);

            for (float delta = - radius; delta <= + radius; delta += 1.0f)
            {
                int32_t coordinate_x = static_cast<int32_t>(x + 0.5f + delta);

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
        }
    }


    void draw_edge_vertical_more(uint8_t* Canvas, BORDERS_T& Borders, VERTICES_T& Vertices, EDGE_T& Edge)
    {
        int32_t channels    = 3;
        float   radius      = 2.0f;
        float   x_initial   = 0.0f;
        float   y_initial   = 0.0f;
        float   x_final     = 0.0f;
        float   y_final     = 0.0f;
        float   x_step      = 0.0f;
        float   y_step      = 1.0f;

        if (Vertices[Edge.IndexA].Y < Vertices[Edge.IndexB].Y)
        {
            x_initial   = Vertices[Edge.IndexA].X;
            x_final     = Vertices[Edge.IndexB].X;
            y_initial   = Vertices[Edge.IndexA].Y;
            y_final     = Vertices[Edge.IndexB].Y;
        }
        else
        {
            x_initial   = Vertices[Edge.IndexB].X;
            x_final     = Vertices[Edge.IndexA].X;
            y_initial   = Vertices[Edge.IndexB].Y;
            y_final     = Vertices[Edge.IndexA].Y;
        }

        float   x = x_initial;
        float   y = y_initial;

        x_step = (x_final - x_initial) / std::abs(y_final - y_initial);

        while (y <= y_final)
        {
            int32_t coordinate_y = static_cast<int32_t>(y + 0.5f);

            for (float delta = - radius; delta <= + radius; delta += 1.0f)
            {
                int32_t coordinate_x = static_cast<int32_t>(x + 0.5f + delta);

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


    void draw_edge_horizontal_more(uint8_t* Canvas, BORDERS_T& Borders, VERTICES_T& Vertices, EDGE_T& Edge)
    {
        int32_t channels    = 3;
        float   radius      = 2.0f;
        float   x_initial   = 0.0f;
        float   y_initial   = 0.0f;
        float   x_final     = 0.0f;
        float   y_final     = 0.0f;
        float   x_step      = 1.0f;
        float   y_step      = 0.0f;

        if (Vertices[Edge.IndexA].X < Vertices[Edge.IndexB].X)
        {
            x_initial   = Vertices[Edge.IndexA].X;
            x_final     = Vertices[Edge.IndexB].X;
            y_initial   = Vertices[Edge.IndexA].Y;
            y_final     = Vertices[Edge.IndexB].Y;
        }
        else
        {
            x_initial   = Vertices[Edge.IndexB].X;
            x_final     = Vertices[Edge.IndexA].X;
            y_initial   = Vertices[Edge.IndexB].Y;
            y_final     = Vertices[Edge.IndexA].Y;
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


    void draw_edge(uint8_t* Canvas, BORDERS_T& Borders, VERTICES_T& Vertices, EDGE_T& Edge)
    {
        int32_t delta_x =   Vertices[Edge.IndexA].X - Vertices[Edge.IndexB].X >= 0 ?
                            Vertices[Edge.IndexA].X - Vertices[Edge.IndexB].X : Vertices[Edge.IndexB].X - Vertices[Edge.IndexA].X;
        int32_t delta_y =   Vertices[Edge.IndexA].Y - Vertices[Edge.IndexB].Y >= 0 ?
                            Vertices[Edge.IndexA].Y - Vertices[Edge.IndexB].Y : Vertices[Edge.IndexB].Y - Vertices[Edge.IndexA].Y;

        if (delta_x == 0)
        {
            draw_edge_vertical_pure(Canvas, Borders, Vertices, Edge);
        }
        else if (delta_y >= delta_x)
        {
            draw_edge_vertical_more(Canvas, Borders, Vertices, Edge);
        }
        else // if (delta_x < delta_y)
        {
            draw_edge_horizontal_more(Canvas, Borders, Vertices, Edge);
        }
    }
}


bool print(
    std::string&    Filename,
    VERTICES_T&     Vertices,
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

    scale(borders, Vertices, Begin, End);

    height = borders.YMax - borders.YMin;
    width  = borders.XMax - borders.XMin;

    canvas = new uint8_t[channels * width * height]{};

    draw_begin(canvas, borders, Begin);
    draw_end  (canvas, borders, End);

    for (IEDGE_T i = Edges.begin(); i != Edges.end(); i++)
    {
        draw_edge(canvas, borders, Vertices, *i);
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
        for (int32_t x = 0; x < width; x++)
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
