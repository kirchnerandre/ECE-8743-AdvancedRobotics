function Edges = get_tangents(ObstacleA, ObstacleB)
    Edges = zeros(4, 3);
    index = 0;

    x_min = ObstacleA(2, 1);
    x_max = ObstacleA(2, 1);
    y_min = ObstacleA(2, 2);
    y_max = ObstacleA(2, 2);

    for a = 2:ObstacleA(1, 1)
        if ObstacleA(a + 1, 1) > x_max
            x_max = ObstacleA(a + 1, 1);
        end

        if ObstacleA(a + 1, 1) < x_min
            x_min = ObstacleA(a + 1, 1);
        end

        if ObstacleA(a + 1, 2) > y_max
            y_max = ObstacleA(a + 1, 2);
        end

        if ObstacleA(a + 1, 2) < y_min
            y_min = ObstacleA(a + 1, 2);
        end
    end

    for b = 1:ObstacleB(1, 1)
        if ObstacleB(b + 1, 1) > x_max
            x_max = ObstacleB(b + 1, 1);
        end

        if ObstacleB(b + 1, 1) < x_min
            x_min = ObstacleB(b + 1, 1);
        end

        if ObstacleB(b + 1, 2) > y_max
            y_max = ObstacleB(b + 1, 2);
        end

        if ObstacleB(b + 1, 2) < y_min
            y_min = ObstacleB(b + 1, 2);
        end
    end

    for a = 1:ObstacleA(1, 1)
        for b = 1:ObstacleB(1, 1)
            m = (ObstacleA(a + 1, 2) - ObstacleB(b + 1, 2)) ...
              / (ObstacleA(a + 1, 1) - ObstacleB(b + 1, 1));

            n =  ObstacleA(a + 1, 2) - ObstacleA(a + 1, 1) * m;

            if ObstacleA(a + 1, 1) == ObstacleB(b + 1, 1)
                point_1 = [ ObstacleA(a + 1, 1) y_min ];
                point_2 = [ ObstacleB(b + 1, 1) y_max ];
            elseif abs(m) > 1
                point_1 = [ (y_min - n) / m     y_min ];
                point_2 = [ (y_max - n) / m     y_max ];
            else
                point_1 = [ x_min               x_min * m + n ];
                point_2 = [ x_max               x_max * m + n ];
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

                for k = 1:index
                    m_old_num   = ObstacleA(Edges(k, 1) + 1, 2) ...
                                - ObstacleB(Edges(k, 2) + 1, 2);
                    m_old_den   = ObstacleA(Edges(k, 1) + 1, 1) ...
                                - ObstacleB(Edges(k, 2) + 1, 1);

                    n_old_num   = m_old_den * ObstacleA(Edges(k, 1) + 1, 2) ...
                                - m_old_num * ObstacleA(Edges(k, 1) + 1, 1);
                    n_old_den   = m_old_den;

                    if (m_new_num * m_old_den ==  m_old_num * m_new_den) ...
                    && (n_new_num * n_old_den ==  n_old_num * n_new_den)
                        duplicated = true;
                        break;
                    end
                end

                if duplicated == false
                    index           = index + 1;
                    Edges(index, :) = [ a b 0 ];
                end
            end
        end
    end
end
