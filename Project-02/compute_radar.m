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

        if fix(obstacle_angles(index_first)) ~= fix(obstacle_angles(index_last))
            angle_min = ceil (min(mod(360 + obstacle_angles(index_first), 360), ...
                                  mod(360 + obstacle_angles(index_last ), 360)));

            angle_max = floor(max(mod(360 + obstacle_angles(index_first), 360), ...
                                  mod(360 + obstacle_angles(index_last ), 360)));

            for j = angle_min:angle_max
                angle       = mod(j + 360 - 1, 360) + 1;

if angle == 49
    [obstacle_angles(index_first) obstacle_angles(index_last) angle_min angle_max]
    angle;
end

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
