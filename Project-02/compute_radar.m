
function RadarData = compute_radar(Obstacles, PositionCurrent, SensorRange)
    step        = 0.5;
    angles      = 0:step:(360 - step);
    steps       = size(angles, 2);
    RadarData   = Inf(1, size(angles, 2));

    for i = 1:size(Obstacles, 2)
        angle       = 180 / pi * atan2(Obstacles(2, i) - PositionCurrent(2), ...
                                       Obstacles(1, i) - PositionCurrent(1));

        index       = mod(round(angle / step), steps) + 1;

        distance    = sqrt((Obstacles(2, i) - PositionCurrent(2)) ^ 2 ...
                         + (Obstacles(1, i) - PositionCurrent(1)) ^ 2);

%        if index >= 118 && index <= 120 && distance < SensorRange
%            [index distance]
%        end

        if (distance < RadarData(index)) && (distance < SensorRange)
            if index == 1
                RadarData(steps)        = distance; 
            else
                RadarData(index - 1)    = distance;
            end

            RadarData(index) = distance;

            if index == steps
                RadarData(1)            = distance; 
            else
                RadarData(index + 1)    = distance;
            end
        end
    end
end
