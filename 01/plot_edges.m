function plot_edges(Edges, Vertices)
    for i = 1:size(Edges, 1)
        plot([Vertices(Edges(i, 1), 1) Vertices(Edges(i, 2), 1)], ...
             [Vertices(Edges(i, 1), 2) Vertices(Edges(i, 2), 2)], 'b')
    end
end
