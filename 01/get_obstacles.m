function [ Obstacles Changed ] = get_obstacles(VertexInitial, VertexFinal, Obstacles)
    Changed     = false;
    points      = 4 * max(abs(VertexInitial(1) - VertexFinal(1)), ...
                          abs(VertexInitial(2) - VertexFinal(2)));

    points_x    = linspace(VertexInitial(1), VertexFinal(1), points);
    points_y    = linspace(VertexInitial(2), VertexFinal(2), points);

    for i = 1:size(Obstacles, 3)
        if Obstacles(1, 2, i) == 0
            [ in, on ] = inpolygon(points_x, ...
                                   points_y, ...
                                   Obstacles(2:end, 1, i), ...
                                   Obstacles(2:end, 2, i));
    
            if max(xor(in, on)) == 1
                Obstacles(1, 2, i)  = 1;
                Changed             = true;
                break;
            end
        end
    end
end
