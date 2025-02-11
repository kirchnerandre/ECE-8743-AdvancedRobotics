function [ Distance Path Edges Vertices ] = visibility_graph(VertexInitial, VertexFinal, Obstacles)
    Vertices    = get_vertices(VertexInitial, VertexFinal, Obstacles);
    Edges       = get_egdes(Vertices, Obstacles);

    [ Path Distance ] = get_path(Vertices, Edges);

    plot_obstacles(Obstacles)

    plot_edges(Edges, Vertices)

    plot_path(Path,  Vertices)
end
