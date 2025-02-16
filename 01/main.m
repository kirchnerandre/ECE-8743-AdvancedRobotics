
close all
clear all
clc

disp('                          vg         tangent         reduced')
disp('Map 1')
compute([   0  16 ], [   16    0 ], 'Map1.txt', [ -2,   18, -2,   18 ],  1)
disp('Map 2')
compute([ 200 200 ], [ 1800 1800 ], 'Map2.txt', [  0, 2000,  0, 2000 ], 20)
disp('Map 3')
compute([  10  10 ], [  180  180 ], 'Map3.txt', [  0,  250,  0,  250 ], 10)

function compute(VertexInitial, VertexFinal, File, Axis, Diameter)
    obstacles = load_obstacles(File);

    for i = 1:size(obstacles, 3)
        obstacles(1, 2, i) = 1;
    end

    [ time_vg distance_vg path_vg edges_vg vertices_vg ] = ...
        compute_vg(VertexInitial, VertexFinal, obstacles);

    plot_data(path_vg, edges_vg, vertices_vg, obstacles, Axis, Diameter)

    for i = 1:size(obstacles, 3)
        obstacles(1, 2, i) = 0;
    end

    [ time_vgr distance_vgr path_vgr edges_vgr vertices_vgr ] = ...
        compute_vgr(VertexInitial, VertexFinal, obstacles);

    plot_data(path_vgr, edges_vgr, vertices_vgr, obstacles, Axis, Diameter)

    for i = 1:size(obstacles, 3)
        obstacles(1, 2, i) = 1;
    end

    [ time_vgt distance_vgt path_vgt edges_vgt vertices_vgt ] = ...
        compute_vgt(VertexInitial, VertexFinal, obstacles);

    plot_data(path_vgt, edges_vgt, vertices_vgt, obstacles, Axis, Diameter)

    out_edges = sprintf("Edges      = %15d %15d %15d", size(edges_vg,  1), ...
                                                       size(edges_vgt, 1), ...
                                                       size(edges_vgr, 1));

    out_dist  = sprintf("Distance   = %15f %15f %15f", distance_vg, ...
                                                       distance_vgt, ...
                                                       distance_vgr);

    out_time  = sprintf("Time       = %15f %15f %15f", time_vg, ...
                                                       time_vgt, ...
                                                       time_vgr);

    disp(out_edges)
    disp(out_dist)
    disp(out_time)
end
