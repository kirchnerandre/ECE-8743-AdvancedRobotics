function [ Time Distance Path Edges Vertices ] = compute_vgt(VertexInitial, ...
                                                             VertexFinal, ...
                                                             Obstacles)
    %
    % This function executes the tangent Visibility Graph algorithm. It
    % creates a list of all vertices in the map, and then create a list of
    % edges containing.
    %
    % 1. All tangent edges between pair of obstacles
    % 2. Edges between VertexInitial and every other vertex in the map
    % 3. Edges between VertexFinal and every other vertex in the map
    % 4. Edges connecting neigboring vertices of the obstacles
    %
    % Discard edges that collide with any obstacle, and then compute the
    % shortest path using the remaining edges
    %
    % compute_vgt(VERTEX_INITIAL, VERTEX_FINAL, OBSTACLES)
    %     VERTICES = get_vertices(VERTEX_INITIAL, VERTEX_FINAL, OBSTACLES)
    %     EDGES    = { get_tangents_all(OBSTACLES) get_tangents_non(VERTICES, OBSTACLES) }
    %     EDGES    = clean_edges(EDGES)
    %     PATH     = get_path(EDGES)
    %

    tic

    Distance    = 0;
    Path        = [];

    Vertices    = get_vertices(VertexInitial, VertexFinal, Obstacles);
    Edges       = [ get_tangents_all(Vertices, Obstacles); ...
                    get_tangents_non(Vertices, Obstacles); ];

    Edges       = clean_edges(Edges, Vertices, Obstacles);

    [ Path Distance ] = get_path(Vertices, Edges);

    Time = toc;
end
