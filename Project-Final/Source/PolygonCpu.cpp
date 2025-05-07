
#include <cstdlib>

#include "PolygonCpu.h"


namespace
{
    bool get_coefficients(float& M, float& N, VERTEX_T& VertexA, VERTEX_T& VertexB)
    {
        if (VertexA.X == VertexB.X)
        {
            return false;
        }
        else
        {
            M = (VertexB.Y - VertexA.Y) / (VertexB.X - VertexA.X);

            N = (VertexA.Y) - M * (VertexA.X);

            return true;
        }

    }


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
                float y_a   = VertexA.Y <= VertexB.Y ? VertexA.Y : VertexB.Y;
                float y_b   = VertexA.Y >  VertexB.Y ? VertexA.Y : VertexB.Y;

                float y_c   = VertexC.Y <= VertexD.Y ? VertexC.Y : VertexD.Y;
                float y_d   = VertexC.Y >  VertexD.Y ? VertexC.Y : VertexD.Y;

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
            float y_a   = VertexA.Y <= VertexB.Y ? VertexA.Y : VertexB.Y;
            float y_b   = VertexA.Y >  VertexB.Y ? VertexA.Y : VertexB.Y;

            float y_c   = VertexC.Y <= VertexD.Y ? VertexC.Y : VertexD.Y;
            float y_d   = VertexC.Y >  VertexD.Y ? VertexC.Y : VertexD.Y;

            float m_cd  = 0.0f;
            float n_cd  = 0.0f;

            get_coefficients(m_cd, n_cd, VertexC, VertexD);

            float y     = m_cd * VertexA.X + n_cd;

            if ((y_a <= y) && (y <= y_b) && (y_c <= y) && (y <= y_d) && (VertexC.X <= VertexA.X) && (VertexA.X <= VertexD.X))
            {
                return true;
            }
            else
            {
                return false;
            }
        }
        else if (VertexC.X == VertexD.X)
        {
            float y_a   = VertexA.Y <= VertexB.Y ? VertexA.Y : VertexB.Y;
            float y_b   = VertexA.Y >  VertexB.Y ? VertexA.Y : VertexB.Y;

            float y_c   = VertexC.Y <= VertexD.Y ? VertexC.Y : VertexD.Y;
            float y_d   = VertexC.Y >  VertexD.Y ? VertexC.Y : VertexD.Y;

            float m_ab  = 0.0f;
            float n_ab  = 0.0f;

            get_coefficients(m_ab, n_ab, VertexA, VertexB);

            float y     = m_ab * VertexC.X + n_ab;

            if ((y_a <= y) && (y <= y_b) && (y_c <= y) && (y <= y_d) && (VertexA.X <= VertexC.X) && (VertexC.X <= VertexB.X))
            {
                return true;
            }
            else
            {
                return false;
            }
        }
        else
        {
            float x_a   = VertexA.X <= VertexB.X ? VertexA.X : VertexB.X;
            float x_b   = VertexA.X >  VertexB.X ? VertexA.X : VertexB.X;

            float x_c   = VertexC.X <= VertexD.X ? VertexC.X : VertexD.X;
            float x_d   = VertexC.X >  VertexD.X ? VertexC.X : VertexD.X;

            float y_a   = VertexA.Y <= VertexB.Y ? VertexA.Y : VertexB.Y;
            float y_b   = VertexA.Y >  VertexB.Y ? VertexA.Y : VertexB.Y;

            float y_c   = VertexC.Y <= VertexD.Y ? VertexC.Y : VertexD.Y;
            float y_d   = VertexC.Y >  VertexD.Y ? VertexC.Y : VertexD.Y;

            float m_ab  = (VertexB.Y - VertexA.Y) / (VertexB.X - VertexA.X);

            float n_ab  = VertexA.Y - VertexA.X * m_ab;

            float m_cd  = (VertexD.Y - VertexC.Y) / (VertexD.X - VertexC.X);

            float n_cd  = VertexC.Y - VertexC.X * m_cd;

            float x     = - (n_cd - n_ab) / (m_cd - m_ab);
            float y     = m_ab * x + n_ab;

            if ((x_a <= x) && (x <= x_b) && (x_c <= x) && (x <= x_d) && (y_a <= y) && (y <= y_b) && (y_c <= y) && (y <= y_d))
            {
                return true;
            }
            else
            {
                return false;
            }
        }
    }


    int32_t cross_product(VERTEX_T& VertexA, VERTEX_T& VertexB, VERTEX_T& VertexC, VERTEX_T& VertexD)
    {
        int32_t x_ab = VertexB.X - VertexA.X;
        int32_t y_ab = VertexB.Y - VertexA.Y;

        int32_t x_cd = VertexC.X - VertexD.X;
        int32_t y_cd = VertexC.Y - VertexD.Y;

        return x_ab * y_cd - x_cd * y_ab;
    }


    bool test_edge_inside_polygon(VERTEX_T& VertexA, VERTEX_T& VertexB, VERTEX_T* Vertices, int32_t PolygonBegin, int32_t PolygonEnd)
    {
        int32_t     vertex_o    = PolygonBegin;

        VERTEX_T    vertex_m    = { (VertexA.X + VertexB.X) / 2, (VertexA.Y + VertexB.Y) / 2};

        for (int32_t vertex_j = vertex_o + 2; vertex_j < PolygonEnd; vertex_j++)
        {
            int32_t vertex_i = vertex_j - 1;

            float oi_x_om = cross_product(Vertices[vertex_o], Vertices[vertex_i], Vertices[vertex_o], vertex_m);
            float ij_x_im = cross_product(Vertices[vertex_i], Vertices[vertex_j], Vertices[vertex_i], vertex_m);
            float jo_x_jm = cross_product(Vertices[vertex_j], Vertices[vertex_o], Vertices[vertex_j], vertex_m);

            if ((oi_x_om >= 0.0f) && (ij_x_im >= 0.0f) && (jo_x_jm >= 0.0f))
            {
                return true;
            }
            else if ((oi_x_om <= 0.0f) && (ij_x_im <= 0.0f) && (jo_x_jm <= 0.0f))
            {
                return true;
            }
        }

        return false;
    }
}


bool test_colision(VERTEX_T* Vertices, NUMBER_T PolygonBegin, NUMBER_T PolygonEnd, EDGE_T& Edge)
{
    for (int32_t i = PolygonBegin; i < PolygonEnd; i++)
    {
        int32_t offset_c = -1;
        int32_t offset_d = -1;

        if (i == PolygonEnd - 1)
        {
            offset_c = i;
            offset_d = PolygonBegin;
        }
        else
        {
            offset_c = i;
            offset_d = i + 1;
        }

        if (test_edges_colision(Vertices[offset_c], Vertices[offset_d], Vertices[Edge.IndexA], Vertices[Edge.IndexB]))
        {
            return true;
        }
    }

    if (test_edge_inside_polygon(Vertices[Edge.IndexA], Vertices[Edge.IndexB], Vertices, PolygonBegin, PolygonEnd))
    {
        return true;
    }

    return false;
}
