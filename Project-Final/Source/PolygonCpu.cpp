
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
            M = static_cast<float>(VertexB.Y - VertexA.Y) / static_cast<float>(VertexB.X - VertexA.X);

            N = static_cast<float>(VertexA.Y) - M * static_cast<float>(VertexA.X);

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
            int32_t y_a = VertexA.Y <= VertexB.Y ? VertexA.Y : VertexB.Y;
            int32_t y_b = VertexA.Y >  VertexB.Y ? VertexA.Y : VertexB.Y;

            int32_t y_c = VertexC.Y <= VertexD.Y ? VertexC.Y : VertexD.Y;
            int32_t y_d = VertexC.Y >  VertexD.Y ? VertexC.Y : VertexD.Y;

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
            int32_t y_a = VertexA.Y <= VertexB.Y ? VertexA.Y : VertexB.Y;
            int32_t y_b = VertexA.Y >  VertexB.Y ? VertexA.Y : VertexB.Y;

            int32_t y_c = VertexC.Y <= VertexD.Y ? VertexC.Y : VertexD.Y;
            int32_t y_d = VertexC.Y >  VertexD.Y ? VertexC.Y : VertexD.Y;

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
            int32_t x_a = VertexA.X <= VertexB.X ? VertexA.X : VertexB.X;
            int32_t x_b = VertexA.X >  VertexB.X ? VertexA.X : VertexB.X;

            int32_t x_c = VertexC.X <= VertexD.X ? VertexC.X : VertexD.X;
            int32_t x_d = VertexC.X >  VertexD.X ? VertexC.X : VertexD.X;

            int32_t y_a = VertexA.Y <= VertexB.Y ? VertexA.Y : VertexB.Y;
            int32_t y_b = VertexA.Y >  VertexB.Y ? VertexA.Y : VertexB.Y;

            int32_t y_c = VertexC.Y <= VertexD.Y ? VertexC.Y : VertexD.Y;
            int32_t y_d = VertexC.Y >  VertexD.Y ? VertexC.Y : VertexD.Y;

            float m_ab  = static_cast<float>(VertexB.Y - VertexA.Y)
                        / static_cast<float>(VertexB.X - VertexA.X);

            float n_ab  = VertexA.Y - VertexA.X * m_ab;

            float m_cd  = static_cast<float>(VertexD.Y - VertexC.Y)
                        / static_cast<float>(VertexD.X - VertexC.X);

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


    bool test_edge_inside_polygon(VERTEX_T& VertexA, VERTEX_T& VertexB, VERTICES_T& Vertices, int32_t VertexBegin, int32_t VertexEnd)
    {
        int32_t vertex_o = VertexBegin;

        for (int32_t vertex_j = vertex_o + 2; vertex_j < VertexEnd; vertex_j++)
        {
            int32_t vertex_i = vertex_j - 1;

            int32_t oi_x_oa = cross_product(Vertices[vertex_o], Vertices[vertex_i], Vertices[vertex_o], VertexA);
            int32_t ij_x_ia = cross_product(Vertices[vertex_i], Vertices[vertex_j], Vertices[vertex_i], VertexA);
            int32_t jo_x_ja = cross_product(Vertices[vertex_j], Vertices[vertex_o], Vertices[vertex_j], VertexA);

            if ((oi_x_oa >= 0) && (ij_x_ia >= 0) && (jo_x_ja >= 0))
            {
                return true;
            }
            else if ((oi_x_oa <= 0) && (ij_x_ia <= 0) && (jo_x_ja <= 0))
            {
                return true;
            }

            int32_t oi_x_ob = cross_product(Vertices[vertex_o], Vertices[vertex_i], Vertices[vertex_o], VertexB);
            int32_t ij_x_ib = cross_product(Vertices[vertex_i], Vertices[vertex_j], Vertices[vertex_i], VertexB);
            int32_t jo_x_jb = cross_product(Vertices[vertex_j], Vertices[vertex_o], Vertices[vertex_j], VertexB);

            if ((oi_x_ob >= 0) && (ij_x_ib >= 0) && (jo_x_jb >= 0))
            {
                return true;
            }
            else if ((oi_x_ob <= 0) && (ij_x_ib <= 0) && (jo_x_jb <= 0))
            {
                return true;
            }
        }

        return false;
    }
}


bool test_colision(VERTICES_T& Vertices, int32_t VertexBegin, int32_t VertexEnd, EDGE_T& Edge)
{

    for (int32_t i = VertexBegin; i < VertexEnd; i++)
    {
        int32_t offset_c = -1;
        int32_t offset_d = -1;

        if (i == VertexEnd - 1)
        {
            offset_c = i;
            offset_d = VertexBegin;
        }
        else
        {
            offset_c = i;
            offset_d = i + 1;
        }

        if (test_edges_colision(Vertices[offset_c], Vertices[offset_d], Edge.VertexA, Edge.VertexB))
        {
            return true;
        }
    }

    if (test_edge_inside_polygon(Edge.VertexA, Edge.VertexB, Vertices, VertexBegin, VertexEnd))
    {
        return true;
    }

    return false;
}
