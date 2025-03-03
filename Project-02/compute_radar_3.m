function RadarData = compute_radar_3(ObstaclesData, ...
                                     ObstaclesLength, ...
                                     PositionCurrent, ...
                                     SensorRange)
    angles                  = 1:360;
    RadarData               = Inf(1, size(angles, 2));

    obstacle_data_angles    = atan2(ObstaclesData(2, :) - PositionCurrent(2), ...
                                    ObstaclesData(1, :) - PositionCurrent(1)) .* 180 ./ pi;

    obstacle_data_distances = sqrt((ObstaclesData(1, :) - PositionCurrent(1)) .^ 2 ...
                                 + (ObstaclesData(2, :) - PositionCurrent(2)) .^ 2);

    index_first = 1;

    for i = 1:size(ObstaclesLength, 2)
        index_last = index_first - 1 + ObstaclesLength(i);

        [ ~, sorted_indexes ] = sort(obstacle_data_angles(index_first:index_last));

        sorted_angles       = obstacle_data_angles   (sorted_indexes);
        sorted_distances    = obstacle_data_distances(sorted_indexes);

        angle_current       = floor(sorted_angles(1));

        for j = angles
            index_before    = -1;
            index_after     = -1;

            for k = 1:size(sorted_angles, 2)
                if sorted_angles(k) < j
                    index_before    = k;
                elseif (index_before ~= -1) && (sorted_angles(k) > j)
                    index_after     = k;
                    [j sorted_angles(index_before) sorted_angles(index_after) ]
                    break;
                end
            end
        end

        index_first = index_first + ObstaclesLength(i);
    end
end
