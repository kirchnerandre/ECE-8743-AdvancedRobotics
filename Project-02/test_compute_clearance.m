close all
clear all
clc

clearance   = 20;
radar_range = 0.5;
position    = [ 5.0; 0.8 ];

[ obstacles_data obstacles_length ] = build_obstacles_1();

radar_data = compute_radar(obstacles_data, obstacles_length, position, radar_range);

radar_data = compute_clearance(radar_data, clearance);

plot_data(obstacles_data, obstacles_length, position, radar_data, radar_range, [ 0 10 0 2 ]);

