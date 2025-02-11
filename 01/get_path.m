function [ Path Distance ] = get_path(Vertices, Edges)
    sources = zeros(2 * size(Edges, 1), 1);
    targets = zeros(2 * size(Edges, 1), 1);
    weights = zeros(2 * size(Edges, 1), 1);

    for i = 1:size(Edges, 1)
        sources(2 * i - 1)  = Edges(i, 1);
        targets(2 * i - 1)  = Edges(i, 2);
        weights(2 * i - 1)  = Edges(i, 3);

        sources(2 * i)      = Edges(i, 2);
        targets(2 * i)      = Edges(i, 1);
        weights(2 * i)      = Edges(i, 3);
    end

    G = graph(sources, targets, weights);

    [ Path Distance ] = shortestpath(G, 1, size(Vertices, 1));
end
