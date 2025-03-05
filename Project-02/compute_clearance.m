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
            j = 1:+1:Clearance;
        elseif radar_status(i) == 2
            j = Clearance:-1:1;
        end

        if radar_status(i) ~= 0
            k = mod(360 + i + j - 1, 360) + 1;
    
            if RadarData(k) == Inf
                RadarData(k) = distance;
            end
        end
    end
end
