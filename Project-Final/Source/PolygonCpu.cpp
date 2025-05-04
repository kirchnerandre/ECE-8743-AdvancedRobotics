
#include <cstdlib>

#include "PolygonCpu.h"


namespace
{
    bool test_edges_colision(VERTEX_T& VertexA, VERTEX_T& VertexB, VERTEX_T& VertexC, VERTEX_T& VertexD)
    {
        if ((VertexA.X == VertexB.X) && (VertexC.X == VertexD.X))
        {
            if (VertexA.X != VertexC.X)
            {
                return false;
            }
            else
            {
                int32_t y_a = VertexA.Y <= VertexB.Y ? VertexA.Y : VertexB.Y;
                int32_t y_b = VertexA.Y >  VertexB.Y ? VertexA.Y : VertexB.Y;

                int32_t y_c = VertexC.Y <= VertexD.Y ? VertexC.Y : VertexD.Y;
                int32_t y_d = VertexC.Y >  VertexD.Y ? VertexC.Y : VertexD.Y;

                if ((y_a <= y_c) && (y_c <= y_b))
                {
                    return true;
                }
                else if ((y_a <= y_d) && (y_d <= y_b))
                {
                    return true;
                }
                else if ((y_c <= y_a) && (y_a <= y_d))
                {
                    return true;
                }
                else if ((y_c <= y_b) && (y_b <= y_d))
                {
                    return true;
                }
                else
                {
                    return false;
                }
            }
        }
        else if (VertexA.X == VertexB.X)
        {

        }
        else if (VertexC.X == VertexD.X)
        {

        }
        else
        {
            float m_ab  = static_cast<float>(VertexB.Y - VertexA.Y)
                        / static_cast<float>(VertexB.X - VertexA.X);

            float n_ab  = VertexA.Y - VertexA.X * m_ab;

            float m_cd  = static_cast<float>(VertexD.Y - VertexC.Y)
                        / static_cast<float>(VertexD.X - VertexC.X);

            float n_cd  = VertexC.Y - VertexC.X * m_cd;

            float x     = - (n_cd - n_ab) / (m_cd - m_ab);

            if ((VertexA.X <= VertexB.X) && (VertexA.X <= x) && (x <= VertexB.X))
            {
                return true;
            }
            else if ((VertexB.X < VertexA.X) && (VertexB.X <= x) && (x <= VertexA.X))
            {
                return true;
            }
            else
            {
                return false;
            }
        }
    }
}


bool test_colision(VERTICES_T& Vertices, int32_t Offset, int32_t Size, EDGE_T& Edge)
{
    for (int32_t i = 0; i < Size; i++)
    {
        int32_t offset_c = Offset + i;
        int32_t offset_d = -1;

        if (i == Size - 1)
        {
            offset_d = Offset;
        }
        else
        {
            offset_d = Offset + i + 1;
        }

        if (test_edges_colision(Edge.VertexA, Edge.VertexB, Vertices[offset_c], Vertices[offset_d]))
        {
            printf("true\n");
            return true;
        }
    }

    printf("false\n");
    return false;
}
