close all
clear all
clc

sensor_range        = 0.5;
position_current    = [ 3.25 3.75 ];
position_final      = [ 5.00 6.00 ];

[ obstacles_data, obstacles_length ] = create_obstacles(position_current, ...
                                                        position_final);

figure
hold on

while true
    radar_data = compute_radar(obstacles_data, position_current, sensor_range);

    plot_data(position_current, position_current, position_final, obstacles_data, obstacles_length, radar_data, sensor_range)

    pause(0.01)
end

