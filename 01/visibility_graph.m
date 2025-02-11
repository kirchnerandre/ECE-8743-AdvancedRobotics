function [ Time Distance ] = visibility_graph(VertexInitial, VertexFinal, Obstacles)
    tic

    vertices    = get_vertices(VertexInitial, VertexFinal, Obstacles);
    edges       = get_egdes(vertices, Obstacles);
    
    [ path Distance ] = get_path(vertices, edges);
    
    plot_obstacles(Obstacles)
    
    plot_edges(edges, vertices)
    
    plot_path(path,  vertices)
    
    Time = toc;
end
