function Edges = get_egdes(Vertices)
    %
    % Creates one edge for each pair of vertices
    %
    % get_egdes(VERTICES)
    %     SIZE  = calculate_size(VERTICES)
    %     EDGES = allocate_memory(SIZE)
    %     INDEX = 0
    %
    %     INDEX_A = { VERTICES(first), ..., VERTICES(last) }
    %         INDEX_B = { VERTICES(INDEX_A + 1), ..., VERTICES(last) }
    %             VERTEX_A      = VERTICES(INDEX_A)
    %             VERTEX_B      = VERTICES(INDEX_B)
    %             INDEX         = INDEX + 1
    %             EDGES(INDEX)  = { VERTEX_A VERTEX_B }
    %
    %     return EDGES
    %

    edges_size  = factorial(size(Vertices,1)) ...
                / factorial(size(Vertices,1) - 2) ...
                / 2;
    Edges       = zeros(cast(edges_size, 'uint32'), 3);
    index       = 0;

    for i = 1:size(Vertices, 1)
        for j = (i + 1):size(Vertices, 1)
            index = index + 1;
            Edges(index, :) = [i j 0];
        end
    end
end
