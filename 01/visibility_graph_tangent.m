function [ Distance Path Edges Vertices ] = visibility_graph_tangent(VertexInitial, VertexFinal, Obstacles)
    Distance = 0;
    Path = [];

    Vertices    = get_vertices(VertexInitial, VertexFinal, Obstacles);
    Edges       = get_tangents_all(Vertices, Obstacles);

    figure
    axis([ -2 18 -2 18 ]);
    axis square;
    hold on
    plot_obstacles(Obstacles)
    plot_edges(Edges, Vertices)

    Edges       = clean_edges(Edges, Vertices, Obstacles);
    
    figure
    axis([ -2 18 -2 18 ]);
    axis square;
    hold on
    plot_obstacles(Obstacles)
    plot_edges(Edges, Vertices)
end
