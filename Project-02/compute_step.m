function [ PositionMiddle PositionBegin ] = compute_step(PositionBegin, PositionFinal, RadarData, StepSize)
    step    = 0.5;
    angles  = 0:step:(360 - step);
    steps   = size(angles, 2);

    angle   = atan2(PositionFinal(2) - PositionBegin(2), ...
                    PositionFinal(1) - PositionBegin(1)) * 180 / pi;

    index   = mod(round(angle / step), steps) + 1;

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

        index_first = -1;
        index_last  = -1;
        indexes     = zeros(1, steps);
    
        for j = 1:steps
            k = mod(i + j, steps) + 1

            if RadarData(k) ~= Inf && index_first == -1
                indexes(k) = 1;
            end
    
            if RadarData(k) == Inf && index_last == -1
                indexes(k - 1) = 1;
            end
   
            if index_first ~= -1 && index_last ~= -1
                index_first = -1;
                index_last  = -1;
            end
        end

        distance = Inf;

        for l = 1:steps
            if indexes(i) == 1
                distance_new = RadarData(l) + sqrt((PositionBegin(1) - PositionFinal(1)) ^ 2 ...
                                                 + (PositionBegin(2) - PositionFinal(2)) ^ 2);

                if (distance_new < distance)
                    distance        = distance_new;
                    PositionBegin   = [cos(angles(l) * pi / 180) * StepSize + PositionBegin(1); ...
                                       sin(angles(l) * pi / 180) * StepSize + PositionBegin(2)];
            
                    PositionMiddle  = PositionBegin;
                end
            end
        end
    end
end
