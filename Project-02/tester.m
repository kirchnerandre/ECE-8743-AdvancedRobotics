close all
clear all
clc

position_begin = [ 3.00 3.00 3.00; ...
                   3.00 6.00 3.00];

position_end   = [ 6.00 5.25 4.00; ...
                   6.00 3.00 6.00];

sensor_ranges   = [ 0.45; 0.50; 0.65 ];
step_sizes      = [ 0.05; 0.10; 0.15 ];
clearance       = 10;
filename        = 'obstacles_1.txt';

for i = 1:length(position_begin)
    for j = 1:length(sensor_ranges)
        for k = 1:length(step_sizes)
            sensor_range    = sensor_ranges(j);
            step_size       = step_sizes   (k);
    
            [ time distance ] = main(position_begin(:, i), ...
                                     position_end  (:, i), ...
                                     sensor_range, ...
                                     step_size, ...
                                     clearance, ...
                                     filename);
            
            [ i sensor_range step_size time distance ]
        end
    end
end
