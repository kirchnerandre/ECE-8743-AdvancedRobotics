close all
clear all
clc

%test_1()
test_2()

function test_1()
    radar_range = 0.5;
    position    = [ 5.0; 0.8 ];

    [ obstacles_data obstacles_length ] = build_obstacles_1();

    radar_data = compute_obstacles(obstacles_data, obstacles_length, position, radar_range);

    plot_data(obstacles_data, obstacles_length, position, radar_data, radar_range);
end

function test_2()
    radar_range = 0.5;
    position    = [ 5.0; 0.8 ];

    [ obstacles_data obstacles_length ] = build_obstacles_2();

    radar_data = compute_obstacles(obstacles_data, obstacles_length, position, radar_range);

    plot_data(obstacles_data, obstacles_length, position, radar_data, radar_range);
end

function [ ObstaclesData ObstaclesLength ] = build_obstacles_1()
    size_1          = [linspace(0.0, 10.0, 50); linspace(1.0, 1.0, 50)];
    ObstaclesData   = [size_1];
    ObstaclesLength = size(ObstaclesData, 2);
end

function [ ObstaclesData ObstaclesLength ] = build_obstacles_2()
    size_1          = [linspace(0.0, 10.0, 50); linspace(1.0, 1.0, 50)];
    size_2          = [linspace(10.0, 0.0, 50); linspace(1.1, 1.1, 50)];
    ObstaclesData   = [size_1 size_2];
    ObstaclesLength = size(ObstaclesData, 2);
end

function plot_data(ObstaclesData, ...
                   ObstaclesLength, ...
                   Position, ...
                   RadarData, ...
                   RadarRange)
    plot_obstacles(ObstaclesData, ObstaclesLength)

    hold on
    axis([ 0 10 0 2 ]);
    axis equal
    set(gca,'XLimMode','manual');
    set(gca,'YLimMode','manual');

    plot_robot(Position)
    plot_radar_range(Position, RadarRange, 1)
    plot_radar_detection(Position, RadarData, 1)
end
