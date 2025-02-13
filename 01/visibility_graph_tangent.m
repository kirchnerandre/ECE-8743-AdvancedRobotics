function [ Distance Path Edges Vertices ] = visibility_graph_tangent(VertexInitial, VertexFinal, Obstacles)
    Distance    = 0;
    Path        = [];

    Vertices    = get_vertices(VertexInitial, VertexFinal, Obstacles);
    Edges       = [ get_tangents_all(Vertices, Obstacles); ...
                    get_tangents_non(Vertices, Obstacles); ];

    Edges       = clean_edges(Edges, Vertices, Obstacles);

    [ Path Distance ] = get_path(Vertices, Edges);
end
