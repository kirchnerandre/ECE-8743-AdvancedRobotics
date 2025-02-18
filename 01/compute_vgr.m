function [ Time Distance Path Edges Vertices ] = compute_rvg(VertexInitial, ...
                                                             VertexFinal, ...
                                                             Obstacles)
    %
    % This function executes my reduced Visibility Graph algorithm. It
    % creates a list of vertices that has only the initial and final
    % vertices in the first iteration. Then computes the shortest path
    % and verifies if it collides with an obstacle. If so, the vertices of
    % that obstacle are added to list of vertices in use, and a new
    % shortest path is computed and verified. This process repeats until a
    % shortest path that does not collide with any obstacle is computed.
    %
    % compute_vgr(VERTEX_INITIAL, VERTEX_FINAL, OBSTACLES)
    %     VERTICES = { VERTEX_INITIAL, VERTEX_FINAL }
    %     READY    = false
    %
    %     while READY == false
    %         READY = true
    %         EDGES = get_egdes(VERTICES)
    %         EDGES = clean_edges(EDGES)
    %         PATH  = get_path(EDGES)
    %
    %         PATH_EDGE = { PATH }
    %             OBSTACLE = { OBSTACLES }
    %
    %             if PATH_EDGE and OBSTACLE collide
    %                 VERTICES = { VERTICES + get_vertices(OBSTACLE) }
    %                 READY    = false
    %                 break
    %
    %     return PATH
    %

    tic

    ready = true;
   
    while ready == true
        ready               = false;
        Vertices            = get_vertices(VertexInitial, VertexFinal, Obstacles);
        Edges               = get_egdes(Vertices);
        Edges               = clean_edges(Edges, Vertices, Obstacles);
        [ Path Distance ]   = get_path(Vertices, Edges);
    
        for i = 2:size(Path, 2)
            [ Obstacles ready ] = get_obstacles(Vertices(Path(i - 1), :), ...
                                                Vertices(Path(i    ), :), ...
                                                Obstacles);
    
            if ready == true
                break;
            end
        end
    end

    Time = toc;
end
