function Vertices = get_vertices(VertexInitial, VertexFinal, Obstacles)
    Vertices            = zeros(sum(Obstacles(1, 1, :)) + 2, 2);
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
