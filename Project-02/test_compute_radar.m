close all
clear all
clc

%test_1()
%test_2()
%test_3()
%test_4()
%test_5()
%test_6()
%test_7()
test_8()

function test_1()
    radar_range = 0.5;
    position    = [ 5.0; 0.8 ];

    [ obstacles_data obstacles_length ] = build_obstacles_1();

    radar_data = compute_radar(obstacles_data, obstacles_length, position, radar_range);

    plot_data(obstacles_data, obstacles_length, position, radar_data, radar_range, [ 0 10 0 2 ]);
end

function test_2()
    radar_range = 0.5;
    position    = [ 5.0; 0.8 ];

    [ obstacles_data obstacles_length ] = build_obstacles_2();

    radar_data = compute_radar(obstacles_data, obstacles_length, position, radar_range);

    plot_data(obstacles_data, obstacles_length, position, radar_data, radar_range, [ 0 10 0 2 ]);
end

function test_3()
    radar_range = 0.5;
    position    = [ 5.0; 0.8 ];

    [ obstacles_data obstacles_length ] = build_obstacles_3();

    radar_data = compute_radar(obstacles_data, obstacles_length, position, radar_range);

    plot_data(obstacles_data, obstacles_length, position, radar_data, radar_range, [ 0 10 0 4 ]);
end

function test_4()
    radar_range = 0.5;
    position    = [ 5.0; 0.8 ];

    [ obstacles_data obstacles_length ] = build_obstacles_4();

    radar_data = compute_radar(obstacles_data, obstacles_length, position, radar_range);

    plot_data(obstacles_data, obstacles_length, position, radar_data, radar_range, [ 0 10 0 2 ]);
end

function test_5()
    radar_range = 0.5;
    position    = [ 5.0; 0.8 ];

    [ obstacles_data obstacles_length ] = build_obstacles_5();

    radar_data = compute_radar(obstacles_data, obstacles_length, position, radar_range);

    plot_data(obstacles_data, obstacles_length, position, radar_data, radar_range, [ 0 10 0 2 ]);
end

function test_6()
    radar_range = 0.5;
    position    = [ 5.0; 0.8 ];

    [ obstacles_data obstacles_length ] = build_obstacles_6();

    radar_data = compute_radar(obstacles_data, obstacles_length, position, radar_range);

    plot_data(obstacles_data, obstacles_length, position, radar_data, radar_range, [ 0 10 0 2 ]);
end

function test_7()
    radar_range = 0.5;
    position    = [ 3.4967; 3.3869 ];

    [ obstacles_data obstacles_length ] = build_obstacles_7();

    radar_data = compute_radar(obstacles_data, obstacles_length, position, radar_range);

    plot_data(obstacles_data, obstacles_length, position, radar_data, radar_range, [ 2 4 3 4 ]);
end

function test_8()
    radar_range = 0.5;
    position    = [ 3.7536; 3.3969 ];

    [ obstacles_data obstacles_length ] = build_obstacles_7();

    radar_data = compute_radar(obstacles_data, obstacles_length, position, radar_range);

    plot_data(obstacles_data, obstacles_length, position, radar_data, radar_range, [ 2 4 3 4 ]);
end

function [ ObstaclesData ObstaclesLength ] = build_obstacles_1()
    side_1          = [linspace(0.0, 10.0, 50); linspace(1.0, 1.0, 50)];
    ObstaclesData   = [side_1];
    ObstaclesLength = size(ObstaclesData, 2);
end

function [ ObstaclesData ObstaclesLength ] = build_obstacles_2()
    side_1          = [linspace(0.0, 10.0, 50); linspace(1.0, 1.0, 50)];
    side_2          = [linspace(10.0, 0.0, 50); linspace(1.1, 1.1, 50)];
    ObstaclesData   = [side_1 side_2];
    ObstaclesLength = size(ObstaclesData, 2);
end

function [ ObstaclesData ObstaclesLength ] = build_obstacles_3()
    side_1          = [linspace(4.0, 5.0, 5); linspace(1.0, 1.0, 5)];
    side_2          = [linspace(5.0, 5.0, 5); linspace(1.0, 1.2, 5)];
    side_3          = [linspace(5.0, 4.0, 5); linspace(1.2, 1.2, 5)];
    side_4          = [linspace(4.0, 4.0, 5); linspace(1.2, 1.0, 5)];
    ObstaclesData   = [side_1 side_2 side_3 side_4];
    ObstaclesLength = size(ObstaclesData, 2);
end

function [ ObstaclesData ObstaclesLength ] = build_obstacles_4()
    side_1          = [linspace(4.0, 4.8, 5); linspace(1.0, 1.0, 5)];
    side_2          = [linspace(4.8, 4.8, 5); linspace(1.0, 1.2, 5)];
    side_3          = [linspace(4.8, 4.0, 5); linspace(1.2, 1.2, 5)];
    side_4          = [linspace(4.0, 4.0, 5); linspace(1.2, 1.0, 5)];
    ObstaclesData   = [side_1 side_2 side_3 side_4];
    ObstaclesLength = size(ObstaclesData, 2);
end

function [ ObstaclesData ObstaclesLength ] = build_obstacles_5()
    side_1          = [linspace(5.2, 5.2, 5); linspace(0.0, 2.0, 5)];
    ObstaclesData   = [side_1];
    ObstaclesLength = size(ObstaclesData, 2);
end

function [ ObstaclesData ObstaclesLength ] = build_obstacles_6()
    side_1          = [linspace(4.0, 4.8, 5); linspace(1.0, 1.0, 5)];
    side_2          = [linspace(4.8, 4.8, 5); linspace(1.0, 1.2, 5)];
    side_3          = [linspace(4.8, 4.0, 5); linspace(1.2, 1.2, 5)];
    side_4          = [linspace(4.0, 4.0, 5); linspace(1.2, 1.0, 5)];
    side_5          = [linspace(5.2, 5.2, 5); linspace(0.5, 1.5, 5)];
    ObstaclesData   = [side_1 side_2 side_3 side_4 side_5];
    ObstaclesLength = [size([side_1 side_2 side_3 side_4], 2) ...
                       size([side_5], 2)];
end

function [ ObstaclesData ObstaclesLength ] = build_obstacles_7()
    n           = 100;
    width       = 2;
    height      = 3;
    left_bottom = [ 3 3 ];
    
    side_1  = [linspace(0.7, 2.5, n); linspace(2.1, 1.3, n)];
    side_2  = [linspace(2.5, 1.8, n); linspace(1.3, 3.2, n)];
    side_3  = [linspace(1.8, 0.7, n); linspace(3.2, 2.1, n)];

    obstacle = [side_1(:, 1:n-1) ...
                side_2(:, 1:n-1) ...
                side_3(:, 1:n-1)];

    obstacle(1, :) = obstacle(1, :) / 10 * width  + left_bottom(1);
    obstacle(2, :) = obstacle(2, :) / 10 * height + left_bottom(2);

    ObstaclesData   = obstacle;

    ObstaclesLength = [ size(obstacle, 2) ];
end

function plot_data(ObstaclesData, ...
                   ObstaclesLength, ...
                   Position, ...
                   RadarData, ...
                   RadarRange, ...
                   PlotAxis)
    clf
    hold on
    axis(PlotAxis)
    axis equal
    set(gca,'XLimMode','manual');
    set(gca,'YLimMode','manual');

    plot_obstacles(ObstaclesData, ObstaclesLength)
    plot_robot(Position)
    plot_radar_range(Position, RadarRange, 1)
    plot_radar_detection(Position, RadarData, 1)
end
