function [ Distance Path Edges Vertices ] = visibility_graph(VertexInitial, ...
                                                             VertexFinal, ...
                                                             Obstacles)

    Vertices    = get_vertices(VertexInitial, VertexFinal, Obstacles);
    Edges       = get_egdes(Vertices);
    Edges       = clean_edges(Edges, Vertices, Obstacles);

    [ Path Distance ] = get_path(Vertices, Edges);
end
