
close all
clear all
clc

tic

vertex_initial      = [4,  4 ];
vertex_final        = [90, 85];

obstacles(:,:,1)    = [4 0; 20 10; 20 30; 60 30; 60 10];
obstacles(:,:,2)    = [4 0; 70 10; 70 50; 90 50; 90 10];
obstacles(:,:,3)    = [4 0; 10 40; 10 60; 50 60; 50 40];
obstacles(:,:,4)    = [4 0; 20 70; 20 90; 80 90; 80 70];

figure()
axis([0 100 0 100])
axis square
hold on

circles(vertex_initial(1), vertex_initial(2), 2, 'facecolor', 'green')
circles(vertex_final  (1), vertex_final  (2), 2, 'facecolor', 'yellow')

plot_obstacles(obstacles)

vertices    = get_vertices(vertex_initial, vertex_final, obstacles);
edges       = get_egdes(vertices, obstacles);

[ path distance ] = get_path(vertices, edges);

plot_obstacles(obstacles)

plot_edges(edges, vertices)

plot_path(path,  vertices)

time = toc;

disp("Time      = " + time);
disp("Distance  = " + distance);
