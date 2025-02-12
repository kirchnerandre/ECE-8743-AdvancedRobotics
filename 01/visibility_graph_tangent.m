function [ Distance Path Edges Vertices ] = visibility_graph_tangent(VertexInitial, VertexFinal, Obstacles)
    vertices        = get_vertices(VertexInitial, VertexFinal, Obstacles);
    vertices_size   = size(vertices, 1);
    edges_index     = 0;
    edges_size      = factorial(vertices_size) / factorial(vertices_size - 2) / 2;
    Edges           = zeros(edges_size, 3);
    offsets         = zeros(size(Obstacles, 3), 1);
    offsets(1)      = 1;

    for i = 2:size(Obstacles, 3)
        offsets(i) = offsets(i - 1) + Obstacles(1, 1, i);
    end

    for i = 1:size(Obstacles, 3)
        for j = (i + 1):size(Obstacles, 3)
            [ vertices_a vertices_b distance ] = get_tangents(Obstacles(:,:,i), Obstacles(:,:,j));

            for k = 1:4
                edges_index = edges_index + 1;
                Edges(edges_index, 1) = [ vertices_a(k) + offsets(i) ];
                Edges(edges_index, 2) = [ vertices_b(k) + offsets(j) ];
                Edges(edges_index, 3) = distance(k);
            end
        end
    end

    [ Path Distance ] = get_path(Vertices, Edges);
end
