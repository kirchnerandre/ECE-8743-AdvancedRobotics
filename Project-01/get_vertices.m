function Vertices = get_vertices(VertexInitial, VertexFinal, Obstacles)
    %
    % Return a list of vertices where the first vertex is the initial
    % vertex, the last vertex is the final vertex, and the vertices between
    % are the vertices of only the flagged obstacles
    %

    length = 0;

    for i = 1:size(Obstacles, 3)
        if Obstacles(1, 2, i) == 1
            length = length + Obstacles(1, 1, i);
        end
    end

    Vertices            = zeros(length, 2);
    index               = 1;
    Vertices(index, :)  = VertexInitial;
    
    for i = 1:size(Obstacles, 3)
        if Obstacles(1, 2, i) == 1
            for j = 2:Obstacles(1, 1, i)+1
                index               = index + 1;
                Vertices(index, :)  = Obstacles(j, :, i);
            end
        end
    end

    index               = index + 1;
    Vertices(index, :)  = VertexFinal;
end
