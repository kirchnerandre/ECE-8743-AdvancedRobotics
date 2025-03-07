close all
clear all
clc


clearance       = 20;
sensor_range    = 0.5;
step_size       = 0.01;
position_begin  = [ 4.8258; 4.9692 ];
position_final  = [ 5.0000; 6.0000 ];

[ obstacles_data obstacles_length ] = create_obstacles(position_begin, ...
                                                       position_final);

figure;
axis([ 2.5 5.5 3.0 6.0 ]);
axis equal
set(gca,'XLimMode','manual');
set(gca,'YLimMode','manual');
hold on

plot_obstacles(obstacles_data, obstacles_length);

plot(position_begin(1), position_begin(2), 'r+', "LineWidth", 2, "MarkerSize", 5)
plot(position_final(1), position_final(2), 'r*', "LineWidth", 2, "MarkerSize", 5)

radar_data = compute_radar(obstacles_data, obstacles_length, position_begin, sensor_range);

radar_data = compute_clearance(radar_data, clearance);

[ position_middle position_begin ] = compute_step_3(position_begin, ...
                                                    position_final, ...
                                                    radar_data, ...
                                                    step_size);

plot_robot(position_begin)

plot_path(position_begin, position_middle, position_final)

plot_radar_range(position_begin, sensor_range, 1.0)

plot_radar_detection(position_begin, radar_data, 1.0)
