close all
clear all
clc

step_size       = 0.01;
sensor_range    = 0.5;
sensor_angle    = 2.0;
sensor_error    = 5;
position_begin  = [ 3.00; 3.00 ];
position_middle = position_begin;
position_final  = [ 5.00; 6.00 ];
left_bottom     = [min(position_begin(1), position_final(1)) ...
                   min(position_begin(2), position_final(2))];
right_top       = [max(position_begin(1), position_final(1)) ...
                   max(position_begin(2), position_final(2))];

[ obstacles_data, obstacles_length, max_distance ] = create_obstacles(position_begin, ...
                                                                      position_final);

assert(sensor_angle > 180 / pi * asin(max_distance / 2 / sensor_range), 'Invalid sensor_angle');


figure('units', 'normalized', 'outerposition', [0 0 1 1])
hold on
axis([left_bottom(1) right_top(1) left_bottom(2) right_top(2)]);
axis equal

while true
    radar_data = compute_radar(obstacles_data, position_begin, sensor_range, sensor_angle);

    plot_data(position_begin, position_middle, position_final, obstacles_data, obstacles_length, radar_data, sensor_range, sensor_angle)

    [ position_middle position_begin ] = compute_step(position_begin, position_final, radar_data, sensor_angle, step_size);

    distance = sqrt((position_final(1) - position_begin(1)) ^ 2 ...
                  + (position_final(2) - position_begin(2)) ^ 2);

    if distance < step_size
        break
    end

    pause(0.01)
end
