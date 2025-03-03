close all
clear all
clc

n               = 50;
obstacle_data   = [linspace(4.0, 6.0, n); linspace(1.0, 1.0, n)];
radar_range     = 0.5;

% test 1
position_1      = [ 5.0, 0.0 ];
radar_data_1    = compute_radar_2(obstacle_data, n, position_1, radar_range);

plot_data(obstacle_data, n, position_1, radar_data_1, radar_range);

% test 2
position_2      = [ 5.0, 0.5 ];
radar_data_2    = compute_radar_2(obstacle_data, n, position_2, radar_range);

plot_data(obstacle_data, n, position_2, radar_data_2, radar_range);

% test 3
position_3      = [ 5.0, 0.8 ];
radar_data_3    = compute_radar_2(obstacle_data, n, position_3, radar_range);

plot_data(obstacle_data, n, position_3, radar_data_3, radar_range);

% test 5
obstacle_data   = [linspace(0.0, 10.0, n); linspace(1.0, 1.0, n)];

position_5      = [ 5.0, 0.8 ];
radar_data_5    = compute_radar_2(obstacle_data, n, position_5, radar_range);

plot_data(obstacle_data, n, position_5, radar_data_5, radar_range);

function plot_data(ObstaclesData, ...
                   ObstaclesLength, ...
                   Position, ...
                   RadarData, ...
                   RadarRange)
    figure;
    hold on
    axis([ 0 10 0 2 ]);
    axis equal
    set(gca,'XLimMode','manual');
    set(gca,'YLimMode','manual');

    plot_obstacles(ObstaclesData, ObstaclesLength)
    plot_robot(Position)
    plot_radar_range(Position, RadarRange, 1)
    plot_radar_detection(Position, RadarData, 1)
end
