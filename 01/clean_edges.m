function Edges = clean_edges(Edges, Vertices, Obstacles)
    %
    % Discards edges that collide with any polynomial.
    %
    % clean_edges(EDGES, VERTICES, OBSTACLES)
    %    SIZE  = get_size(EDGES)
    %    CLEAN = allocate_memory(SIZE)
    %    INDEX = 0
    %
    %    EDGE in { EDGES }
    %    COORDINATES_X = { EDGE.VERTEX_INITIAL.X, ..., EDGE.VERTEX_FINAL.X }
    %    COORDINATES_Y = { EDGE.VERTEX_INITIAL.Y, ..., EDGE.VERTEX_FINAL.Y }
    %
    %    COLLISION = false
    %
    %    OBSTACLE in { OBSTACLES }
    %        if (COORDINATES_X and OBSTACLE collide) or (COORDINATES_Y and OBSTACLE collide)
    %           COLLISION = true
    %
    %    if COLLISION is false
	%        EDGE.LENGTH  = calculate_length(EDGE.VERTEX_INITIAL, EDGE.VERTEX_FINAL)
    %        INDEX        = INDEX + 1
    %   	 CLEAN(INDEX) = EDGE
    %
    %    return CLEAN(1:INDEX)
    %

    edges = zeros(size(Edges));
    index = 0;

    for i = 1:size(Edges, 1)
        points = 4 * max(abs(Vertices(Edges(i, 1), 1) - Vertices(Edges(i, 2), 1)), ...
                         abs(Vertices(Edges(i, 1), 2) - Vertices(Edges(i, 2), 2)));

        points_x = linspace(Vertices(Edges(i, 1), 1), Vertices(Edges(i, 2), 1), points);
        points_y = linspace(Vertices(Edges(i, 1), 2), Vertices(Edges(i, 2), 2), points);

        collision = false;

        for j = 1:size(Obstacles, 3)
            if Obstacles(1, 2, j) == 1
                [ in, on ]  = inpolygon(points_x, ...
                                        points_y, ...
                                        Obstacles(2:(Obstacles(1, 1, j) + 1), 1, j), ...
                                        Obstacles(2:(Obstacles(1, 1, j) + 1), 2, j));

                if max(xor(in, on)) == 1
                    collision = true;
                    break
                end
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
