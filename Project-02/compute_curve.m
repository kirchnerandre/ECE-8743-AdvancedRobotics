
function Curve = compute_curve(Obstacles, PositionCurrent, SensorRange)
    step    = 0.5;
    angles  = 0:step:(360 - step);
    steps   = size(angles, 2);
    Curve   = Inf(1, size(angles, 2));

    for o = 1:size(Obstacles, 2)
        angle       = 180 / pi * atan2(Obstacles(2, o) - PositionCurrent(2), ...
                                       Obstacles(1, o) - PositionCurrent(1));

        index       = mod(round(angle / step), steps) + 1;

        distance    = norm(Obstacles(2, o) - PositionCurrent(2), ...
                           Obstacles(1, o) - PositionCurrent(1));

        if (distance < Curve(index)) && (distance < SensorRange)
            Curve(index) = distance;
        end
    end
end
