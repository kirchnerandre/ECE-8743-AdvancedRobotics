function [ ObstaclesData ObstaclesLength ] = build_obstacles()
    n           = 100;
    width       = 2;
    height      = 3;
    left_bottom = [ 3 3 ];

    side_1      = [linspace(0.7, 2.5, n); linspace(2.1, 1.3, n)];
    side_2      = [linspace(2.5, 1.8, n); linspace(1.3, 3.2, n)];
    side_3      = [linspace(1.8, 0.7, n); linspace(3.2, 2.1, n)];

    side_4      = [linspace(5.0, 5.2, n); linspace(3.0, 1.9, n)];
    side_5      = [linspace(5.2, 6.9, n); linspace(1.9, 1.8, n)];
    side_6      = [linspace(6.9, 7.1, n); linspace(1.8, 3.1, n)];
    side_7      = [linspace(7.1, 6.3, n); linspace(3.1, 4.1, n)];
    side_8      = [linspace(6.3, 5.0, n); linspace(4.1, 3.0, n)];

    obstacle_1  = [side_1(:, 1:n-1) ...
                   side_2(:, 1:n-1) ...
                   side_3(:, 1:n  )];

    obstacle_2  = [side_4(:, 1:n-1) ...
                   side_5(:, 1:n-1) ...
                   side_6(:, 1:n-1) ...
                   side_7(:, 1:n-1) ...
                   side_8(:, 1:n  )];

    obstacle_1(1, :) = obstacle_1(1, :) / 10 * width  + left_bottom(1);
    obstacle_1(2, :) = obstacle_1(2, :) / 10 * height + left_bottom(2);

    obstacle_2(1, :) = obstacle_2(1, :) / 10 * width  + left_bottom(1);
    obstacle_2(2, :) = obstacle_2(2, :) / 10 * height + left_bottom(2);


    ObstaclesData   = [ obstacle_1 obstacle_2 ];

    ObstaclesLength = [ size(obstacle_1, 2) size(obstacle_2, 2) ];
end

