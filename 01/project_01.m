
close all
clear all
clc

vertex_initial      = [4,  4 ];
vertex_final        = [90, 85];

obstacles1(:,:,1) = [4 0; 20 10; 20 40; 60 40; 60 10];
obstacles1(:,:,2) = [4 0; 50 60; 50 80; 70 80; 70 60];

[ time distance ] = visable_graph(vertex_initial, vertex_final, obstacles1);

disp("***")
disp("Environment   = 1");
disp("Time          = " + time);
disp("Distance      = " + distance);

obstacles2(:,:,1)    = [4 0; 20 10; 20 30; 60 30; 60 10];
obstacles2(:,:,2)    = [4 0; 70 10; 70 50; 90 50; 90 10];
obstacles2(:,:,3)    = [4 0; 10 40; 10 60; 50 60; 50 40];
obstacles2(:,:,4)    = [4 0; 20 70; 20 90; 80 90; 80 70];

[ time distance ] = visable_graph(vertex_initial, vertex_final, obstacles2);

disp("***")
disp("Environment   = 2");
disp("Time          = " + time);
disp("Distance      = " + distance);

obstacles3(:,:,1) = [4 0; 20 10; 20 30; 40 30; 40 10];
obstacles3(:,:,2) = [4 0; 70 10; 70 30; 90 30; 90 10];
obstacles3(:,:,3) = [4 0; 10 40; 10 60; 30 60; 30 40];
obstacles3(:,:,4) = [4 0; 20 70; 20 90; 40 90; 40 70];
obstacles3(:,:,5) = [4 0; 50 5 ; 50 25; 65 25; 65 5 ];
obstacles3(:,:,6) = [4 0; 40 40; 40 60; 75 60; 75 40];
obstacles3(:,:,7) = [4 0; 10 70; 10 90; 15 90; 15 70];
obstacles3(:,:,8) = [4 0; 50 65; 50 90; 70 90; 70 65];
obstacles3(:,:,9) = [4 0; 80 35; 80 75; 95 75; 95 35];

[ time distance ] = visable_graph(vertex_initial, vertex_final, obstacles3);

disp("***")
disp("Environment   = 3");
disp("Time          = " + time);
disp("Distance      = " + distance);
