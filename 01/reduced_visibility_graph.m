function [ Distance Path Edges Vertices ] = reduced_visibility_graph(VertexInitial, VertexFinal, Obstacles)
    changed = true;
   
    while changed == true
        changed             = false;
        Vertices            = get_vertices(VertexInitial, VertexFinal, Obstacles);
        Edges               = get_egdes(Vertices, Obstacles);
        [ Path Distance ]   = get_path(Vertices, Edges);
    
        for i = 2:size(Path, 2)
            [ Obstacles changed ] = get_obstacles(Vertices(Path(i - 1), :), Vertices(Path(i), :), Obstacles);
    
            if changed == true
                break;
            end
        end
    end
end
