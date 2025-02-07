function Vertices = get_vertices(VertexInitial, VertexFinal, Obstacles)
    vertices_size = 2; % VertexInitial + VertexFinal 
    
    for i = 1:size(Obstacles, 3)
        vertices_size = vertices_size + Obstacles(1, 1, i);
    end
    
    Vertices = zeros(vertices_size, 2);
    
    Vertices(vertices_size, :) = VertexFinal;
    
    for i = 1:size(Obstacles, 3)
        for j = 2:Obstacles(1, 1, i)+1
            vertices_size = vertices_size - 1;
            Vertices(vertices_size, :) = Obstacles(j, :, i);
        end
    end
    
    Vertices(1, :) = VertexInitial;
end
