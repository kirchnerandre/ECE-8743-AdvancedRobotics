function [ Time Distance ] = visable_graph(VertexInitial, VertexFinal, Obstacles)
    tic

    figure()
    axis([0 100 0 100])
    axis square
    hold on
    
    plot_circle(VertexInitial(1), VertexInitial(2), 2, 'facecolor', 'green')
    plot_circle(VertexFinal  (1), VertexFinal  (2), 2, 'facecolor', 'yellow')
    
    vertices    = get_vertices(VertexInitial, VertexFinal, Obstacles);
    edges       = get_egdes(vertices, Obstacles);
    
    [ path Distance ] = get_path(vertices, edges);
    
    plot_obstacles(Obstacles)
    
    plot_edges(edges, vertices)
    
    plot_path(path,  vertices)
    
    Time = toc;
end
