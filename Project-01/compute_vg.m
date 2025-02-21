function [ Time Distance Path Edges Vertices ] = compute_vg(VertexInitial, ...
                                                            VertexFinal, ...
                                                            Obstacles)
    %
    % This function executes the standard Visibility Graph algorithm to find
    % the shortest path between the initial and final vertices. The
    % algorithm first creates a list of all vertices in the map, it
    % creates a list of all edges between every pair of vertices, then it
    % discards all edges that collide with any obstacle, and finally
    % compute the shortest path using the remaining edges
    %
    % compute_vg(VERTEX_INITIAL, VERTEX_FINAL, OBSTACLES)
    %     VERTICES = get_vertices(VERTEX_INITIAL, VERTEX_FINAL, OBSTACLES)
    %     EDGES    = get_egdes(VERTICES)
    %     EDGES    = clean_edges(EDGES, VERTICES, OBSTACLES)
    %     PATH     = get_path(VERTICES, EDGES)
    %
    %     return PATH
    %

    tic

    Vertices            = get_vertices(VertexInitial, VertexFinal, Obstacles);
    Edges               = get_egdes(Vertices);
    Edges               = clean_edges(Edges, Vertices, Obstacles);

    [ Path Distance ]   = get_path(Vertices, Edges);

    Time                = toc;
end
