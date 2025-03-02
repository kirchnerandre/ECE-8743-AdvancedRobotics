%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   TANGENTBUG.M
%   ECE8743 Advanced Robotics
%   Date:   Spring 2024
%   Description:    Implement TangentBug path planning algorithm.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [ ObstaclesData ObstaclesLength MaxDistance ] = create_obstacles(PositionStart, PositionDestiny)
    MaxDistance = 0;
    n           = 100;
    width       = abs(PositionDestiny(1) - PositionStart(1));
    height      = abs(PositionDestiny(2) - PositionStart(2));

    left_bottom = [ min(PositionStart(1), PositionDestiny(1));
                    min(PositionStart(2), PositionDestiny(2))];
    
    side_1  = [linspace(0.7, 2.5, n); linspace(2.1, 1.3, n)];
    side_2  = [linspace(2.5, 1.8, n); linspace(1.3, 3.2, n)];
    side_3  = [linspace(1.8, 0.7, n); linspace(3.2, 2.1, n)];

    side_4  = [linspace(5.0, 5.2, n); linspace(3.0, 1.9, n)];
    side_5  = [linspace(5.2, 6.9, n); linspace(1.9, 1.8, n)];
    side_6  = [linspace(6.9, 7.1, n); linspace(1.8, 3.1, n)];
    side_7  = [linspace(7.1, 6.3, n); linspace(3.1, 4.1, n)];
    side_8  = [linspace(6.3, 5.0, n); linspace(4.1, 3.0, n)];
        
    side_9  = [linspace(5.0, 6.3, n); linspace(7.2, 7.4, n)];
    side_10 = [linspace(6.3, 7.9, n); linspace(7.4, 6.0, n)];
    side_11 = [linspace(7.9, 9.0, n); linspace(6.0, 6.5, n)];
    side_12 = [linspace(9.0, 9.0, n); linspace(6.5, 7.6, n)];
    side_13 = [linspace(9.0, 6.8, n); linspace(7.6, 9.3, n)];
    side_14 = [linspace(6.8, 4.5, n); linspace(9.3, 9.1, n)];
    side_15 = [linspace(4.5, 5.0, n); linspace(9.1, 7.2, n)];
    
    obs_1 = [side_1(:, 1:n-1) ...
             side_2(:, 1:n-1) ...
             side_3(:, 1:n-1)];

    obs_2 = [side_4(:, 1:n-1) ...
             side_5(:, 1:n-1) ...
             side_6(:, 1:n-1) ...
             side_7(:, 1:n-1) ...
             side_8(:, 1:n-1)];

    obs_3 = [side_9( :, 1:n-1) ...
             side_10(:, 1:n-1) ...
             side_11(:, 1:n-1) ...
             side_12(:, 1:n-1) ...
             side_13(:, 1:n-1) ...
             side_14(:, 1:n-1) ...
             side_15(:, 1:n-1)];  

    obs_1(1, :) = obs_1(1, :) / 10 * width  + left_bottom(1);
    obs_1(2, :) = obs_1(2, :) / 10 * height + left_bottom(2);
    obs_2(1, :) = obs_2(1, :) / 10 * width  + left_bottom(1);
    obs_2(2, :) = obs_2(2, :) / 10 * height + left_bottom(2);
    obs_3(1, :) = obs_3(1, :) / 10 * width  + left_bottom(1);
    obs_3(2, :) = obs_3(2, :) / 10 * height + left_bottom(2);

    ObstaclesData   = [obs_1 obs_2 obs_3];

    ObstaclesLength = [ size(obs_1, 2) size(obs_2, 2) size(obs_3, 2) ];

    MaxDistance     = max(get_max_distance(side_1),  MaxDistance);
    MaxDistance     = max(get_max_distance(side_2),  MaxDistance);
    MaxDistance     = max(get_max_distance(side_3),  MaxDistance);
    MaxDistance     = max(get_max_distance(side_5),  MaxDistance);
    MaxDistance     = max(get_max_distance(side_6),  MaxDistance);
    MaxDistance     = max(get_max_distance(side_7),  MaxDistance);
    MaxDistance     = max(get_max_distance(side_8),  MaxDistance);
    MaxDistance     = max(get_max_distance(side_9),  MaxDistance);
    MaxDistance     = max(get_max_distance(side_10), MaxDistance);
    MaxDistance     = max(get_max_distance(side_11), MaxDistance);
    MaxDistance     = max(get_max_distance(side_12), MaxDistance);
    MaxDistance     = max(get_max_distance(side_13), MaxDistance);
    MaxDistance     = max(get_max_distance(side_14), MaxDistance);
    MaxDistance     = max(get_max_distance(side_15), MaxDistance);



%    ObstaclesData   = side_1(:, 1:n-1);

%    ObstaclesLength = [ size(ObstaclesData, 2) ];

%    MaxDistance     = max(get_max_distance(side_1),  0);

end

function MaxDistance = get_max_distance(Side)
    a           = Side(:, 1);
    b           = Side(:, end);
    ab          = sqrt((a(1) - b(1)) ^ 2 + (a(2) - b(2)) ^ 2);

    if size(Side, 2) > 1
        MaxDistance = ab / (size(Side, 2) - 1);
    else
        MaxDistance = 0;
    end
end
