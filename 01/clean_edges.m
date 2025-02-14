function Edges = clean_edges(Edges, Vertices, Obstacles)
    edges = zeros(size(Edges));
    index = 0;

    for i = 1:size(Edges, 1)
        points = 4 * max(abs(Vertices(Edges(i, 1), 1) - Vertices(Edges(i, 2), 1)), ...
                         abs(Vertices(Edges(i, 1), 2) - Vertices(Edges(i, 2), 2)));

        points_x = linspace(Vertices(Edges(i, 1), 1), Vertices(Edges(i, 2), 1), points);
        points_y = linspace(Vertices(Edges(i, 1), 2), Vertices(Edges(i, 2), 2), points);

        collision = false;

        for j = 1:size(Obstacles, 3)
            [ in, on ]  = inpolygon(points_x, ...
                                    points_y, ...
                                    Obstacles(2:(Obstacles(1, 1, j) + 1), 1, j), ...
                                    Obstacles(2:(Obstacles(1, 1, j) + 1), 2, j));

            if max(xor(in, on)) == 1
                collision = true;
                break
            end
        end

        if collision == false
            index          = index + 1;
            edges(index,1) = Edges(i,1);
            edges(index,2) = Edges(i,2);
            edges(index,3) = sqrt((Vertices(Edges(i,1),1) - Vertices(Edges(i,2),1))^2 ...
                                + (Vertices(Edges(i,1),2) - Vertices(Edges(i,2),2))^2);
        end
    end

    Edges = edges(1:index, :);
end
