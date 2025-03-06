function RadarData = compute_radar(ObstaclesData, ...
                                   ObstaclesLength, ...
                                   PositionCurrent, ...
                                   SensorRange)

    index_first = 1;
    RadarData   = Inf(1, 360);

    for i = 1:size(ObstaclesLength, 2)
        index_last = index_first - 1 + ObstaclesLength(i);

        RadarData = compute_obstacle(ObstaclesData(:, index_first:index_last), ...
                                     PositionCurrent, ...
                                     SensorRange, ...
                                     RadarData);

        index_first = index_first + ObstaclesLength(i);
    end
end

function RadarData = compute_obstacle(ObstaclesData, ...
                                      PositionCurrent, ...
                                      SensorRange, ...
                                      RadarData)
    obstacle_angles = atan2(ObstaclesData(2, :) - PositionCurrent(2), ...
                            ObstaclesData(1, :) - PositionCurrent(1)) .* 180 ./ pi;

    for i = 1:size(obstacle_angles, 2)
        index_first = i;
        index_last  = mod(i, size(obstacle_angles, 2)) + 1;

        angle_first = obstacle_angles(index_first);
        angle_last  = obstacle_angles(index_last);

        if abs(angle_first - angle_last) > 180
            if angle_first < 0
                angle_first = angle_first + 360;
            elseif angle_last < 0
                angle_last = angle_last + 360;
            end
        end

        angle_min = min(angle_first, angle_last);
        angle_max = max(angle_first, angle_last);

[ angle_min angle_max ]
        if fix(angle_min) ~= fix(angle_max)
            for j = ceil(angle_min):floor(angle_max)
                angle       = mod(j + 360 - 1, 360) + 1;
    
                distance    = compute_distance(ObstaclesData(:, index_first), ...
                                               ObstaclesData(:, index_last), ...
                                               PositionCurrent, ...
                                               angle);
    
                if distance < SensorRange && distance < RadarData(angle)
                    RadarData(angle) = distance;
                end
            end
        end
    end
end

function Distance = compute_distance(PointA, PointB, PointO, Angle)
    m_oz    = tan(Angle * pi / 180);

    n_oz    =  PointO(2) - PointO(1) * m_oz;

    m_ab    = (PointA(2) - PointB(2)) / (PointA(1) - PointB(1));

    n_ab    =  PointA(2) - PointA(1) * m_ab;

    if PointA(1) == PointB(1) && mod(Angle, 180) == 90
        x = PointA(1);

        y = min(PointA(2), PointB(2));
    elseif PointA(1) == PointB(1)
        x = PointA(1);

        y = x * m_oz + n_oz;
    elseif mod(Angle, 180) == 90
        x = PointO(1);

        y = x * m_ab + n_ab;
    else
        x = - (n_oz - n_ab) / (m_oz - m_ab);

        y = x * m_oz + n_oz;
    end

    Distance = sqrt((x - PointO(1)) ^ 2 + (y - PointO(2)) ^ 2);
end
