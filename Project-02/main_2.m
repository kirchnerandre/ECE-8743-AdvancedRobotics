close all
clear all
clc

step_size           = 0.01;
sensor_range        = 0.5;
position_begin      = [ 3.00 3.00 ];
position_middle     = position_begin;
position_final      = [ 5.00 6.00 ];

[ obstacles_data, obstacles_length ] = create_obstacles(position_begin, ...
                                                        position_final);

figure
hold on

while true
    radar_data = compute_radar(obstacles_data, position_begin, sensor_range);

    plot_data(position_begin, position_middle, position_final, obstacles_data, obstacles_length, radar_data, sensor_range)

    [ position_middle position_begin ] = compute_step(position_begin, position_final, radar_data, step_size);

    pause(0.01)

    distance = sqrt((position_final(1) - position_begin(1)) ^ 2 ...
                  + (position_final(2) - position_begin(2)) ^ 2);

    if distance < step_size
        break
    end
end
