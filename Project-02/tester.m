close all
clear all
clc

position_begin  = [ 3.00; 3.00 ];
position_end    = [ 5.00; 6.00 ];
sensor_range    = 0.5;
step_size       = 0.01;
clearance       = 10;
filename        = 'obstacles_1.txt';

[ time distance ] = main(position_begin, ...
                         position_end, ...
                         sensor_range, ...
                         step_size, ...
                         clearance, ...
                         filename);

[ time distance ]
