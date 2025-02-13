function Edges = get_tangents_all(Vertices, Obstacles)
    obstacles_size  = size(Obstacles, 3);
    tangents_size   = 4 * cast(factorial(obstacles_size)/factorial(obstacles_size - 2)/2, 'uint32');
    Edges           = zeros(tangents_size, 3);
    edges_index     = 0;

    offset_a        = 1;

    for a = 1:size(Obstacles, 3)
        offset_b = offset_a + Obstacles(1, 1, a);

        for b = (a + 1):size(Obstacles, 3)
            edges = get_tangents(Obstacles(:,:,a), Obstacles(:,:,b));

            Edges(edges_index + 1, :) = [ (edges(1, 1) + offset_a) (edges(1, 2) + offset_b) 0 ];
            Edges(edges_index + 2, :) = [ (edges(2, 1) + offset_a) (edges(2, 2) + offset_b) 0 ];
            Edges(edges_index + 3, :) = [ (edges(3, 1) + offset_a) (edges(3, 2) + offset_b) 0 ];
            Edges(edges_index + 4, :) = [ (edges(4, 1) + offset_a) (edges(4, 2) + offset_b) 0 ];

            edges_index = edges_index   + 4;
            offset_b    = offset_b      + Obstacles(1, 1, b);
        end

        offset_a = offset_a + Obstacles(1, 1, a);
    end
end
