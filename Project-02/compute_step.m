function [ PositionMiddle PositionBegin ] = compute_step(PositionBegin, ...
                                                         PositionEnd, ...
                                                         RadarData, ...
                                                         StepSize)
    angle_direct = round(atan2(PositionEnd(2) - PositionBegin(2), ...
                               PositionEnd(1) - PositionBegin(1)) * 180 / pi);

    if angle_direct == 0
        angle_direct = 360;
    elseif angle_direct < 0
        angle_direct = angle_direct + 360;
    end

    if RadarData(angle_direct) == Inf
        position_begin  = PositionBegin + StepSize * [ cos(angle_direct * pi / 180); ...
                                                       sin(angle_direct * pi / 180) ];

        PositionMiddle  = position_begin;
    else
        distance = Inf;

        for i = 1:360
            j = mod(i, 360) + 1;

            if RadarData(i) == Inf && RadarData(j) ~= Inf
                angle_indirect  = j;
            elseif RadarData(i) ~= Inf && RadarData(j) == Inf
                angle_indirect  = i;
            else
                continue
            end

            distance_new = compute_distance(PositionBegin, ...
                                            PositionEnd, ...
                                            RadarData, ...
                                            angle_indirect);

            if distance_new < distance
                distance        = distance_new;
                value_sin       = sin(angle_indirect * pi / 180);
                value_cos       = cos(angle_indirect * pi / 180);

                PositionMiddle  = PositionBegin + ...
                                  RadarData(angle_indirect) * [ value_cos; value_sin ];

                position_begin  = PositionBegin + ...
                                  StepSize                  * [ value_cos; value_sin ];
            end
        end
    end

    PositionBegin = position_begin;
end

function Distance = compute_distance(PositionBegin, PositionEnd, RadarData, Angle)
    Position_middle = PositionBegin + [cos(Angle * pi / 180); ...
                                       sin(Angle * pi / 180)] .* RadarData(Angle);

    Distance = sqrt((Position_middle(1) - PositionBegin(1)) ^ 2 ...
                  + (Position_middle(2) - PositionBegin(2)) ^ 2) ...
             + sqrt((Position_middle(1) - PositionEnd(1)  ) ^ 2 ...
                  + (Position_middle(2) - PositionEnd(2)  ) ^ 2);
end

function Angle = use_angle(AngleCurrent, AnglePrevious, AngleDiffMax)
    if AnglePrevious < 0
        Angle = AngleCurrent;
    else
        angle_diff = AngleCurrent - AnglePrevious;

        if angle_diff < -180
            angle_diff = angle_diff + 360;
        elseif angle_diff > 180
            angle_diff = angle_diff - 360;
        end

        if angle_diff < - AngleDiffMax
            Angle = AnglePrevious - AngleDiffMax;
        elseif angle_diff > AngleDiffMax
            Angle = AnglePrevious + AngleDiffMax;
        else
            Angle = AngleCurrent;
        end
    end
end
