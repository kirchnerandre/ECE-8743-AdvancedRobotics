function Edges = get_tangents_all(Vertices, Obstacles)
    %
    % Return the list of all tangent edges between all pair of obstacles
    %
    % get_tangents_all(OBSTACLES)
    %     SIZE  = calculate_tangents_size(OBSTACLES)
    %     EDGES = allocate_memory(SIZE)
    %     INDEX = 0
    %
    %     INDEX_A = { 1, ..., get_size(OBSTACLES) }
    %         INDEX_B = { (INDEX_A + 1), ..., get_size(OBSTACLES) }
    %             INDEX            = INDEX + 4
    %             TANGENTS         = get_tangents(OBSTACLES(INDEX_A),
    %                                             OBSTACLES(INDEX_B))
    %             EDGES(INDEX - 3) = TANGENTS(1)
    %             EDGES(INDEX - 2) = TANGENTS(2)
    %             EDGES(INDEX - 1) = TANGENTS(3)
    %             EDGES(INDEX - 0) = TANGENTS(4)
    %
    %      return EDGES
    %

    obstacles_size  = size(Obstacles, 3);
    tangents_size   = 4 * factorial(obstacles_size) / factorial(obstacles_size - 2) / 2;
    Edges           = zeros(cast(tangents_size, 'uint32'), 3);
    edges_index     = 0;

    offset_a        = 1;

    for a = 1:size(Obstacles, 3)
        offset_b = offset_a + Obstacles(1, 1, a);

        for b = (a + 1):size(Obstacles, 3)
            edges = get_tangents(Obstacles(:,:,a), Obstacles(:,:,b));

            Edges(edges_index + 1, :) = [ (edges(1, 1) + offset_a) ...
                                          (edges(1, 2) + offset_b) 0 ];
            Edges(edges_index + 2, :) = [ (edges(2, 1) + offset_a) ...
                                          (edges(2, 2) + offset_b) 0 ];
            Edges(edges_index + 3, :) = [ (edges(3, 1) + offset_a) ...
                                          (edges(3, 2) + offset_b) 0 ];
            Edges(edges_index + 4, :) = [ (edges(4, 1) + offset_a) ...
                                          (edges(4, 2) + offset_b) 0 ];

            edges_index = edges_index   + 4;
            offset_b    = offset_b      + Obstacles(1, 1, b);
        end

        offset_a = offset_a + Obstacles(1, 1, a);
    end
end
