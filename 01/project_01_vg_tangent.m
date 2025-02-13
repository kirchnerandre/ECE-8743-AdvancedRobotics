
close all
clear all
clc

compute([   0  16 ], [   16    0 ], 'Map1.txt', -2,   18, -2,   18,  1)
compute([ 200 200 ], [ 1800 1800 ], 'Map2.txt',  0, 2000,  0, 2000, 50)
compute([  10  10 ], [  180  180 ], 'Map3.txt',  0,  250,  0,  250, 10)

function compute(VertexInitial, VertexFinal, File, XMin, XMax, YMin, YMax, Diameter)
    obstacles = load_obstacles(File);

    tic

    for i = 1:size(obstacles, 3)
        obstacles(1, 2, i) = 1;
    end

    figure();
    axis([XMin XMax YMin YMax]);
    axis square;
    hold on

    plot_circle(VertexInitial(1), VertexInitial(2), Diameter, 'facecolor', 'green')
    plot_circle(VertexFinal  (1), VertexFinal  (2), Diameter, 'facecolor', 'yellow')

    plot_obstacles(obstacles)

    [ distance path edges vertices ] = visibility_graph_tangent(VertexInitial,  ...
                                                                VertexFinal,    ...
                                                                obstacles);

    plot_edges(edges, vertices)
    plot_path (path,  vertices)

    disp("Distance      = " + distance);
    disp("Time          = " + toc);
end
