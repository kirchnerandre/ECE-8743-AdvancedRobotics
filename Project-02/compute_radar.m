
function RadarData = compute_radar(Obstacles, PositionCurrent, SensorRange)
    step        = 0.5;
    angles      = 0:step:(360 - step);
    steps       = size(angles, 2);
    RadarData   = Inf(1, size(angles, 2));

    for i = 1:size(Obstacles, 2)
        angle       = 180 / pi * atan2(Obstacles(2, i) - PositionCurrent(2), ...
                                       Obstacles(1, i) - PositionCurrent(1));

        index       = mod(round(angle / step), steps) + 1;

        distance    = norm(Obstacles(2, i) - PositionCurrent(2), ...
                           Obstacles(1, i) - PositionCurrent(1));

        if (distance < RadarData(index)) && (distance < SensorRange)
            RadarData(index) = distance;
        end
    end
end
