function [ PositionMiddle PositionBegin ] = compute_step(PositionBegin, PositionFinal, RadarData, SensorAngle, StepSize)
    angles  = 0:SensorAngle:(360 - SensorAngle);
    steps   = size(angles, 2);

    angle   = atan2(PositionFinal(2) - PositionBegin(2), ...
                    PositionFinal(1) - PositionBegin(1)) * 180 / pi;

    index   = mod(round(angle / SensorAngle), steps) + 1;

    if RadarData(index) == Inf
        PositionBegin   = [cos(angle * pi / 180) * StepSize + PositionBegin(1); ...
                           sin(angle * pi / 180) * StepSize + PositionBegin(2)];

        PositionMiddle  = PositionBegin;
    else
        for i = 1:steps
            if RadarData(i) == Inf
                break;
            end
        end

        state   = false;
        indexes = zeros(1, steps);

        for j = 1:steps
            k = mod(i + j - 1, steps) + 1;

            if RadarData(k) ~= Inf && state == false
                indexes(k)  = 1;
                state       = true;
            end
    
            if RadarData(k) == Inf && state == true
                if k == 1
                    indexes(steps)  = 1;
                else
                    indexes(k - 1)  = 1;
                end

                state = false;
            end
        end

        distance = Inf;

        for l = 1:steps
            if indexes(l) == 1
                distance_new = RadarData(l) + sqrt((PositionBegin(1) - PositionFinal(1)) ^ 2 ...
                                                 + (PositionBegin(2) - PositionFinal(2)) ^ 2);

                if (distance_new < distance)
                    distance            = distance_new;
                    PositionCandidate   = [cos(angles(l) * pi / 180) * StepSize + PositionBegin(1); ...
                                           sin(angles(l) * pi / 180) * StepSize + PositionBegin(2)];
                end
            end
        end

        PositionMiddle  = PositionCandidate;
        PositionBegin   = PositionCandidate;
    end
end
