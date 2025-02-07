function Edges = get_egdes(Vertices, Obstacles)
    edges_size  = factorial(size(Vertices, 1)) / factorial(size(Vertices, 1) - 2) / 2;
    Edges       = zeros(edges_size, 3);
    index       = 0;

    for i = 1:size(Vertices, 1)
        for j = (i + 1):size(Vertices, 1)
            points = 4 * max(abs(Vertices(i, 1) - Vertices(j, 1)), ...
                             abs(Vertices(i, 2) - Vertices(j, 2)));

            points_x = linspace(Vertices(i, 1), Vertices(j, 1), points);
            points_y = linspace(Vertices(i, 2), Vertices(j, 2), points);

            valid = true;

            for k = 1:size(Obstacles, 3)
                [ in, on ] = inpolygon(points_x, ...
                                       points_y, ...
                                       Obstacles(2:end, 1, k), ...
                                       Obstacles(2:end, 2, k));

                if max(xor(in, on))
                    valid = false;
                    break
                end
            end

            if valid
                index = index + 1;
                Edges(index, :) = [i j sqrt((Vertices(i, 1) - Vertices(j, 1)) ^ 2 ...
                                          + (Vertices(i, 2) - Vertices(j, 2)) ^ 2)];
            end
        end
    end

    Edges = Edges(1:index, :);
end
