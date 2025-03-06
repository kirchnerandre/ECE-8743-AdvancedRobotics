function RadarData = compute_clearance(RadarData, Clearance)
    radar_status    = zeros(1, 360);

    for i = 1:360
        angle_this  = i;
        angle_next  = mod(i, 360) + 1;

        if RadarData(angle_this) ~= Inf && RadarData(angle_next) == Inf
            radar_status(angle_this) = 1;
        elseif RadarData(angle_this) == Inf && RadarData(angle_next) ~= Inf
            radar_status(angle_next) = 2;
        end
    end

    for i = 1:360
        distance = RadarData(i);

        if radar_status(i) == 1
            a = mod(360 + i,                 360) + 1;
            b = mod(360 + i + Clearance - 1, 360) + 1;
        elseif radar_status(i) == 2
            a = mod(360 + i - 2 - Clearance, 360) + 1;
            b = mod(360 + i - 2,             360) + 1;
        else
            continue
        end

        if a > b
            b = b + 360;
        end

        for k = a:b
            l = mod(k - 1, 360) + 1;

            if RadarData(l) == Inf
                RadarData(l) = distance;
            end
        end
    end
end
