function [ Time Distance Path Edges Vertices ] = compute_vg(VertexInitial, ...
                                                            VertexFinal, ...
                                                            Obstacles)

    tic

    Vertices            = get_vertices(VertexInitial, VertexFinal, Obstacles);
    Edges               = get_egdes(Vertices);
    Edges               = clean_edges(Edges, Vertices, Obstacles);

    [ Path Distance ]   = get_path(Vertices, Edges);

    Time                = toc;
end
