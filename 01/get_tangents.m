function [ VerticesA VerticesB ] = get_tangents(ObstacleA, ...
                                                ObstacleB, ...
                                                XMin, ...
                                                XMax, ...
                                                YMin, ...
                                                YMax)
    VerticesA = [];
    VerticesB = [];

    for a = 1:ObstacleA(1, 1)
        for b = 1:ObstacleB(1, 1)
            m = (ObstacleA(a + 1, 2) - ObstacleB(b + 1, 2)) ...
              / (ObstacleA(a + 1, 1) - ObstacleB(b + 1, 1));

            n =  ObstacleA(a + 1, 2) - ObstacleA(a + 1, 1) * m;

            if ObstacleA(a + 1, 1) == ObstacleB(b + 1, 1)
                point_1 = [ ObstacleA(a + 1, 1) YMin ];
                point_2 = [ ObstacleB(b + 1, 1) YMax ];
            elseif abs(m) > 1
                point_1 = [ (YMin - n) / m      YMin ];
                point_2 = [ (YMax - n) / m      YMax ];
            else
                point_1 = [ XMin                XMin * m + n ];
                point_2 = [ XMax                XMax * m + n ];
            end

            points          = 4 * max(abs(point_1(1) - point_2(1)), ...
                                      abs(point_1(2) - point_2(2)));

            points_x        = linspace(point_1(1), point_2(1), points);
            points_y        = linspace(point_1(2), point_2(2), points);

            [ in_a, on_a ]  = inpolygon(points_x, ...
                                        points_y, ...
                                        ObstacleA(2:(ObstacleA(1, 1) + 1), 1), ...
                                        ObstacleA(2:(ObstacleA(1, 1) + 1), 2));
    
            [ in_b, on_b ]  = inpolygon(points_x, ...
                                        points_y, ...
                                        ObstacleB(2:(ObstacleB(1, 1) + 1), 1), ...
                                        ObstacleB(2:(ObstacleB(1, 1) + 1), 2));

            if max(max(xor(in_a, on_a), xor(in_b, on_b))) == 0
                m_new_num   = ObstacleA(a + 1, 2) - ObstacleB(b + 1, 2);
                m_new_den   = ObstacleA(a + 1, 1) - ObstacleB(b + 1, 1);

                n_new_num   = m_new_den * ObstacleA(a + 1, 2) ...
                            - m_new_num * ObstacleA(a + 1, 1);
                n_new_den   = m_new_den;

                duplicated  = false;

                for k = 1:size(VerticesA, 1)
                    m_old_num   = VerticesA(k, 2) - VerticesB(k, 2);
                    m_old_den   = VerticesA(k, 1) - VerticesB(k, 1);

                    n_old_num   = m_old_den * VerticesA(k, 2) ...
                                - m_old_num * VerticesA(k, 1);
                    n_old_den   = m_old_den;

                    if (m_new_num * m_old_den ==  m_old_num * m_new_den) ...
                    && (n_new_num * n_old_den ==  n_old_num * n_new_den)
                        duplicated = true;
                        break;
                    end
                end

                if duplicated == false
                    plot(points_x, points_y)

                    VerticesA = [ VerticesA; ObstacleA(a + 1, :) ];
                    VerticesB = [ VerticesB; ObstacleB(b + 1, :) ];
                end
            end
        end
    end
end
