function PositionNext = compute_move(PositionFinal, PositionCurrent, RadarData, StepSize)
    angle       = atan2(PositionFinal(2) - PositionCurrent(2), ...
                        PositionFinal(1) - PositionCurrent(1));
    distance    = Inf;
    step        = 0.5;
    angles      = (0:step:(360 - step)) .* pi / 180;
    steps       = size(angles, 2);
    index_first = -1;
    index_last  = -1;

    for i = 1:steps
        if RadarData(i) ~= Inf && index_first == -1
            index_first = i;
        end

        if RadarData(i) == Inf && index_last == -1
            index_last  = i - 1;
        end

        if index_first ~= -1 && index_last ~= -1
            distance_first = norm(RadarData(index_first) * cos(angles(index_first)) - PositionCurrent(1), ...
                                  RadarData(index_first) * sin(angles(index_first)) - PositionCurrent(2)) ...
                             norm(RadarData(index_first) * cos(angles(index_first)) - PositionFinal  (1), ...
                                  RadarData(index_first) * sin(angles(index_first)) - PositionFinal  (2));

            if distance_first < distance
                distance   = distance_first;
                angle      = angles(index_first);
            end

            distance_last  = norm(RadarData(index_last)  * cos(angles(index_last))  - PositionCurrent(1), ...
                                  RadarData(index_last)  * sin(angles(index_last))  - PositionCurrent(2)) ...
                             norm(RadarData(index_last)  * cos(angles(index_last))  - PositionFinal  (1), ...
                                  RadarData(index_last)  * sin(angles(index_last))  - PositionFinal  (2));

            if distance_last < distance
                distance   = distance_last;
                angle      = angles(index_last);
            end

            index_first = -1;
            index_last  = -1;
        end
    end

    PositionFinal = PositionCurrent + [ StepSize * cos(angle) tepSize * sin(angle) ];
end
