
#include "../Headers/Types.h"

namespace AlgorithPackageWrapping
{
    void compute(VERTICES_T& VerticesOutput, VERTICES_T& VerticesInput)
    {
        if (VerticesInput.size() > 0u)
        {
            size_t index_o = 0u;
            size_t index_i = 0u;

            for (size_t i = 1u; i < VerticesInput.size(); i++)
            {
                if (VerticesInput[i].X < VerticesInput[index_o].X)
                {
                    index_o = i;
                    index_i = i;
                }
            }

            VerticesOutput.push_back(VerticesInput[index_i]);

            bool    angle_status    = false;
            size_t  index_j         = 0u;

            for (size_t j = 0u; j < VerticesInput.size(); j++)
            {
                if (j == index_i)
                {
                    continue;
                }
                else if (!angle_status)
                {
                    angle_status    = true;
                    index_j         = j;
                }
                else if ((VerticesInput[j].Y - VerticesInput[index_i].Y) * (VerticesInput[index_j].X - VerticesInput[index_i].X) >
                         (VerticesInput[j].X - VerticesInput[index_i].X) * (VerticesInput[index_j].Y - VerticesInput[index_i].Y))
                {
                    index_j = j;
                }
            }

            while (index_j != index_o)
            {
                VerticesOutput.push_back(VerticesInput[index_j]);

                size_t index_k = 0u;

                for (size_t k = 1u; k < VerticesInput.size(); k++)
                {
                    if ((VerticesInput[k]      .X - VerticesInput[index_j].X) * (VerticesInput[index_i].X - VerticesInput[index_j].X) + (VerticesInput[k]      .Y - VerticesInput[index_j].Y) * (VerticesInput[index_i].Y - VerticesInput[index_j].Y)
                      < (VerticesInput[index_k].X - VerticesInput[index_j].X) * (VerticesInput[index_i].X - VerticesInput[index_j].X) + (VerticesInput[index_k].Y - VerticesInput[index_j].Y) * (VerticesInput[index_i].Y - VerticesInput[index_j].Y))
                    {
                        index_k = k;
                    }
                }

                index_i = index_j;
                index_j = index_k;
            }
        }
    }
}