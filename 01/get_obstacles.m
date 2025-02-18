function [ Obstacles Changed ] = get_obstacles(VertexInitial, VertexFinal, Obstacles)
    %
    % Verifies it the edge connection VertexInitial and VertexFinal collide
    % with any obstacle. If so, the obstacle will be flagged.
    %
    % get_obstacles(VERTEX_INITIAL, VERTEX_FINAL, Obstacles)
    %     COORDINATES_X = { VERTEX_INITIAL.X, ..., VERTEX_FINAL.X }
    %     COORDINATES_Y = { VERTEX_INITIAL.Y, ..., VERTEX_FINAL.Y }
    %
    %     COLLISION     = false
    %
    %     OBSTACLE in { OBSTACLES }
    %         if COLLISION.FLAGED == false
    %             if (COORDINATES_X and OBSTACLE collide)
    %             or (COORDINATES_Y and OBSTACLE collide)
    %                COLLISION.FLAGED = true
    %                return
    %

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
