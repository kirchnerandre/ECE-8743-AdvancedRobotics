function [ ObstaclesData ObstaclesLength ] = build_obstacles_7()
    side_1          = [linspace(0.0, 10.0, 50); linspace(1.0, 1.0, 50)];
    ObstaclesData   = [side_1];
    ObstaclesLength = size(ObstaclesData, 2);
end

