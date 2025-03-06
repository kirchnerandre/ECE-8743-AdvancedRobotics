close all
clear all
clc


clearance       = 20;
radar_range     = 0.5;
step_size       = 0.01;
position_begin  = [ 3.6170; 3.5450 ];
position_end    = [ 5.0000; 6.0000 ];

[ obstacles_data obstacles_length ] = build_obstacles();

radar_data = compute_radar(obstacles_data, obstacles_length, position_begin, radar_range);

radar_data = compute_clearance(radar_data, clearance);

plot_data(obstacles_data, obstacles_length, position_begin, radar_data, radar_range, [ 3.0 4.5 3.0 4.5 ])

[ position_middle position_begin ] = compute_step_2(position_begin, position_end, radar_data, step_size);

plot_data(obstacles_data, obstacles_length, position_begin, radar_data, radar_range, [ 3.0 4.5 3.0 4.5 ])

plot_route(position_begin, position_middle, position_end)

