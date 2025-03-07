function [ PositionMiddle PositionBegin ] = compute_step_3(PositionBegin, PositionFinal, RadarData, StepSize)
    angle_direct = round(atan2(PositionFinal(2) - PositionBegin(2), PositionFinal(1) - PositionBegin(1)) * 180 / pi);

    if angle_direct == 0
        angle_direct = 360;
    elseif angle_direct < 0
        angle_direct = angle_direct + 360;
    end

    if RadarData(angle_direct) == Inf
        PositionBegin   = PositionBegin + StepSize * [ cos(angle_direct * pi / 180); ...
                                                       sin(angle_direct * pi / 180) ];

        PositionMiddle  = PositionBegin;
    else
        distance = Inf;

        for i = 1:360
            j = mod(i, 360) + 1;

            if RadarData(i) == Inf && RadarData(j) ~= Inf
                distance_new    = compute_distance(PositionBegin, PositionFinal, RadarData, j);
                angle_indirect  = j;
            elseif RadarData(i) ~= Inf && RadarData(j) == Inf
                distance_new    = compute_distance(PositionBegin, PositionFinal, RadarData, i);
                angle_indirect  = i;
            else
                continue
            end

            if distance_new < distance
                distance        = distance_new;

                PositionMiddle  = PositionBegin + RadarData(angle_indirect) * [ cos(angle_indirect * pi / 180); ...
                                                                                sin(angle_indirect * pi / 180) ];

                PositionBegin   = PositionBegin + StepSize                  * [ cos(angle_indirect * pi / 180); ...
                                                                                sin(angle_indirect * pi / 180) ];
            end
        end
    end
end

function Distance = compute_distance(PositionBegin, PositionFinal, RadarData, Angle)
    Position_middle = PositionBegin + [sin(Angle * pi / 180); ...
                                       cos(Angle * pi / 180)] .* RadarData(Angle);

    Distance = sqrt((Position_middle(1) - PositionBegin(1)) ^ 2 ...
                  + (Position_middle(2) - PositionBegin(2)) ^ 2) ...
             + sqrt((Position_middle(1) - PositionFinal(1)) ^ 2 ...
                  + (Position_middle(2) - PositionFinal(2)) ^ 2);
end
