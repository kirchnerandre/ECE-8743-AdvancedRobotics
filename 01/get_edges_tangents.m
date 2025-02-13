function Edges = get_edges_tangents(Vertices, Obstacles, XMin, XMax, YMin, YMax)
    obstacles_size  = size(Obstacles, 3);
    tangents_size   = cast(factorial(obstacles_size)/factorial(obstacles_size - 2)/2, 'uint32');
    Edges           = zeros(4 * tangents_size, 3);
    edges_index     = 0;

    offset_a        = 1;

    for a = 1:size(Obstacles, 3)
        offset_b = offset_a + Obstacles(1, 1, a);

        for b = (a + 1):size(Obstacles, 3)
            [ vertices_a vertices_b ] = get_tangents(Obstacles(:,:,a), Obstacles(:,:,b), XMin, XMax, YMin, YMax);

            edges_index = edges_index + 4;

            Edges(edges_index    ) = [ (vertices_a(1) + offset_a) (vertices_b(1) + offset_b) 0 ];
            Edges(edges_index + 1) = [ (vertices_a(2) + offset_a) (vertices_b(2) + offset_b) 0 ];
            Edges(edges_index + 2) = [ (vertices_a(3) + offset_a) (vertices_b(3) + offset_b) 0 ];
            Edges(edges_index + 3) = [ (vertices_a(4) + offset_a) (vertices_b(4) + offset_b) 0 ];

            offset_b = offset_b + Obstacles(1, 1, b);
        end

        offset_a = offset_a + Obstacles(1, 1, a);
    end
end
