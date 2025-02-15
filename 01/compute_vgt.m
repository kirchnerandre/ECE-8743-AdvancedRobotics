function [ Time Distance Path Edges Vertices ] = compute_vgt(VertexInitial, ...
                                                             VertexFinal, ...
                                                             Obstacles)
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
