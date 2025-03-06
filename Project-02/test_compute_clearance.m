close all
clear all
clc

%test_7()
%test_9()
test_10()

function test_7()
    clearance   = 20;
    radar_range = 0.5;
    position    = [ 3.7536; 3.3969 ];

    [ obstacles_data obstacles_length ] = build_obstacles_7();

    radar_data = compute_radar(obstacles_data, obstacles_length, position, radar_range);

    plot_data(obstacles_data, obstacles_length, position, radar_data, radar_range, [ 2 4 3 4 ]);

    radar_data = compute_clearance(radar_data, clearance);

    plot_data(obstacles_data, obstacles_length, position, radar_data, radar_range, [ 2 4 3 4 ]);
end

function test_9()
    clearance   = 20;
    radar_range = 0.5;
    position    = [ 3.6170; 3.5450 ];

    [ obstacles_data obstacles_length ] = build_obstacles_7();

    radar_data = compute_radar(obstacles_data, obstacles_length, position, radar_range);

    plot_data(obstacles_data, obstacles_length, position, radar_data, radar_range, [ 3.0 4.5 3.0 4.5 ]);

    radar_data = compute_clearance(radar_data, clearance);

    plot_data(obstacles_data, obstacles_length, position, radar_data, radar_range, [ 3.0 4.5 3.0 4.5 ]);
end

function test_10()
    clearance   = 20;
    radar_range = 0.5;
    position    = [ 3.5365; 3.3480 ];

    [ obstacles_data obstacles_length ] = build_obstacles_9();

    radar_data = compute_radar(obstacles_data, obstacles_length, position, radar_range);

    plot_data(obstacles_data, obstacles_length, position, radar_data, radar_range, [ 3.0 4.5 3.0 4.5 ]);

    radar_data = compute_clearance(radar_data, clearance);

    plot_data(obstacles_data, obstacles_length, position, radar_data, radar_range, [ 3.0 4.5 3.0 4.5 ]);
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

function [ ObstaclesData ObstaclesLength ] = build_obstacles_9()
    n           = 100;
    width       = 2;
    height      = 3;
    left_bottom = [ 3 3 ];
    
    side_4      = [linspace(5.0, 5.2, n); linspace(3.0, 1.9, n)];
    side_5      = [linspace(5.2, 6.9, n); linspace(1.9, 1.8, n)];
    side_6      = [linspace(6.9, 7.1, n); linspace(1.8, 3.1, n)];
    side_7      = [linspace(7.1, 6.3, n); linspace(3.1, 4.1, n)];
    side_8      = [linspace(6.3, 5.0, n); linspace(4.1, 3.0, n)];

    obstacle    = [side_4(:, 1:n-1) ...
                   side_5(:, 1:n-1) ...
                   side_6(:, 1:n-1) ...
                   side_7(:, 1:n-1) ...
                   side_8(:, 1:n  )];

    obstacle(1, :) = obstacle(1, :) / 10 * width  + left_bottom(1);
    obstacle(2, :) = obstacle(2, :) / 10 * height + left_bottom(2);


    ObstaclesData   = [ obstacle ];

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
