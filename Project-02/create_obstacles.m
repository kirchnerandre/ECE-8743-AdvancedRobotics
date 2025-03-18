function [ ObstaclesData ObstaclesLength ] = create_obstacles_2(Filename)
    ObstaclesData   = [];
    ObstaclesLength = [];

    data = load(Filename);

    for i = 1:2:size(data, 1)
        obstacle_length = 0;

        for j = 2:3:(3 * data(i, 1))
            position_first  = [ data(i,     j); ...
                                data(i + 1, j)];

            position_last   = [ data(i,     j + 1); ...
                                data(i + 1, j + 1)];

            steps           = data(i,      j + 2);

            positions = [ linspace(position_first(1), position_last(1), steps); ...
                          linspace(position_first(2), position_last(2), steps) ];

            ObstaclesData = [ ObstaclesData positions(:, 1:(steps - 1)) ];

            obstacle_length = obstacle_length + size(positions, 2) - 1;
        end

        ObstaclesLength = [ ObstaclesLength obstacle_length ];
    end
end
