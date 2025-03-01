
function RadarData = compute_radar(Obstacles, PositionCurrent, SensorRange, SensorAngle)
    angles      = 0:SensorAngle:(360 - SensorAngle);
    steps       = size(angles, 2);
    RadarData   = Inf(1, size(angles, 2));

    for i = 1:size(Obstacles, 2)
        angle       = 180 / pi * atan2(Obstacles(2, i) - PositionCurrent(2), ...
                                       Obstacles(1, i) - PositionCurrent(1));

        index       = mod(round(angle / SensorAngle), steps) + 1;

        distance    = sqrt((Obstacles(2, i) - PositionCurrent(2)) ^ 2 ...
                         + (Obstacles(1, i) - PositionCurrent(1)) ^ 2);

        if (distance < RadarData(index)) && (distance < SensorRange)
            RadarData(index) = distance;
        end
    end
end
