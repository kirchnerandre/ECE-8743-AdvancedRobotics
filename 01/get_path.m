function [ Path Distance ] = get_path(Vertices, Edges)
    %
    % Uses the Vertices and Edges to create a Matlab graph that is used to
    % compute the shortest path between the first and last vertex of
    % Vertices
    %
    % get_path(VERTICES, EDGES)
    %     SOURCES = allocate_memory(2 * get_size(EDGES))
    %     TARGETS = allocate_memory(2 * get_size(EDGES))
    %     LENGTH  = allocate_memory(2 * get_size(EDGES))
    %
    %     INDEX = { 1, ...., get_size(EDGES) }
    %         SOURCES(2 * INDEX - 1) = EDGES(INDEX).VERTEX_INITIAL
    %         TARGETS(2 * INDEX - 1) = EDGES(INDEX).VERTEX_FINAL
    %         LENGTH (2 * INDEX - 1) = EDGES(INDEX).LENGTH
    %
    %         SOURCES(2 * INDEX - 0) = EDGES(INDEX).VERTEX_FINAL
    %         TARGETS(2 * INDEX - 0) = EDGES(INDEX).VERTEX_INITIAL
    %         LENGTH (2 * INDEX - 0) = EDGES(INDEX).LENGTH
    %
    %     GRAPH = build_graph(SOURCES, TARGETS, LENGTH)
    % 
    %     PATH  = calculate_path(GRAPH)
    % 
    %     return PATH
    %

    Path        = [];
    Distance    = 0;

    if size(Edges, 1) > 0
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
end
