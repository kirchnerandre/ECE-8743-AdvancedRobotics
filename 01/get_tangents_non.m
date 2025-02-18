function Edges = get_tangents_non(Vertices, Obstacles)
    %
    % return a list of edges containing the following 3 types of edges.
    % 1. Edges between the first vertex of Vertices and all vertices in the
    % obstacles. The first vertex of Vertices is the initial vertex
    % 2. Edges between the last vertex of Vertices and all vertices in the
    % obstacles. The last vertex of Vertices is the final vertex
    % 3. Edges connecting neighboring vertices of the obstacles
    %
    % get_tangents_non(VERTICES, OBSTACLES)
    %     SIZE       = 3 * get_number_of_vertices(OBSTACLES)
    %     EDGES      = allocate_memory(SIZE)
    %     EDGE_INDEX = 0
    %
    %     OBSTACLE = {OBSTACLES}
    %         VERTEX = { OBSTACLE.VERTICES }
    %             EDGE_INDEX        = EDGE_INDEX + 1
    %             EDGES(EDGE_INDEX) = { VERTICES(first) VERTEX }
    %
    %     OBSTACLE = {OBSTACLES}
    %         VERTEX = { OBSTACLE.VERTICES }
    %             INDEX             = INDEX + 1
    %             EDGES(EDGE_INDEX) = { VERTICES(last) VERTEX }
    %
    %     OBSTACLE = {OBSTACLES}
    %         VERTEX_INDEX = { 1, ..., get_size(OBSTACLE.VERTICES) }
    %             EDGE_INDEX = EDGE_INDEX + 1
    %
    %             if VERTEX_INDEX == 1
    %                 EDGES(INDEX) = { OBSTACLE.VERTICES(last)         OBSTACLE.VERTICES(first) }
    %             else
    %                 EDGES(INDEX) = { OBSTACLE.VERTICES(VERTEX_INDEX) OBSTACLE.VERTICES(VERTEX_INDEX + 1) }
    %

    length  = 3 * sum(Obstacles(1, 1, :));
    Edges   = zeros(length, 3);
    initial = 1;
    final   = size(Vertices, 1);
    index   = 0;
    offset  = 1;

    for i = 1:size(Obstacles, 3)
        for j = 1:Obstacles(1, 1, i)
            index           = index + 1;
            Edges(index, :) = [ initial (j + offset) 0 ];
        end

        offset = offset + Obstacles(1, 1, i);
    end

    offset  = 1;

    for i = 1:size(Obstacles, 3)
        for j = 1:Obstacles(1, 1, i)
            index           = index + 1;
            Edges(index, :) = [ final   (j + offset) 0 ];
        end

        offset = offset + Obstacles(1, 1, i);
    end

    offset  = 1;

    for i = 1:size(Obstacles, 3)
        for j = 1:Obstacles(1, 1, i)
            index               = index + 1;

            if j ~= 1
                Edges(index, :) = [ (j + offset - 1) (j + offset)                  0 ];
            else
                Edges(index, :) = [ (1 + offset)     (Obstacles(1, 1, i) + offset) 0 ];
            end
        end

        offset = offset + Obstacles(1, 1, i);
    end
end
