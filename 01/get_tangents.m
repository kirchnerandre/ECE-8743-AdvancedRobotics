function [ VerticesA VerticesB Distance ] = get_tangents(ObstacleA, ObstacleB)
    VerticesA   = zeros(4, 1);
    VerticesB   = zeros(4, 1);
    Distance    = zeros(4, 1);
    index       = 0;

    for i = 1:ObstacleA(1, 1)
        for j = 1:ObstacleB(1, 1)
            [ intersect_a intersect_b ] = test_edge(ObstacleA, ObstacleB, ObstacleA(i+1, :), ObstacleB(j+1, :));

            if (intersect_a == false) && (intersect_b == false)
                index = index + 1;

                VerticesA(index) = i;
                VerticesB(index) = j;
                Distance (index) = sqrt((ObstacleA(i + 1, 1) - ObstacleB(i + 1, 1)) ^ 2 ...
                                      + (ObstacleA(i + 1, 2) - ObstacleB(i + 1, 2)) ^ 2);
            end
        end
    end
end

function [ IntersectA IntersectB ] = test_edge(ObstacleA, ObstacleB, VertexA, VertexB)
    m = (VertexA(2) - VertexB(2)) / (VertexA(1) - VertexB(1));
    n =  VertexA(2) - VertexA(1) * m;

    if m > 1
        vertex_a = [ 0              n ];
        vertex_b = [ 100            (100 * m + n)];
    else
        vertex_a = [ -n/m           0 ];
        vertex_b = [  (100 - n)/m   100 ];
    end

    points = 4 * max(abs(vertex_a(1) - vertex_b(1)), abs(vertex_a(2) - vertex_b(2)));

    points_x = linspace(vertex_a(1), vertex_b(1), points);
    points_y = linspace(vertex_a(2), vertex_b(2), points);

    [ in, on ] = inpolygon(points_x, points_y, ObstacleA(2:end, 1), ObstacleA(2:end, 2));
    
    if max(xor(in, on))
        IntersectA = false;
    else
        IntersectA = true;
    end

    [ in, on ] = inpolygon(points_x, points_y, ObstacleB(2:end, 1), ObstacleB(2:end, 2));
    
    if max(xor(in, on))
        IntersectB = false;
    else
        IntersectB = true;
    end
end
