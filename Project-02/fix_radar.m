function RadarDataNew = fix_radar(RadarDataOld, SensorAngle, SensorError)
    angles          = 0:SensorAngle:(360 - SensorAngle);
    steps           = size(angles, 2);
    RadarDataNew    = Inf(1, steps);

    for i = 1:steps
        distance = Inf;

        for j = -floor(SensorError / 2):floor(SensorError / 2)
            k = mod(i + j - 1, steps) + 1;

            if RadarDataOld(k) < distance
                distance = RadarDataOld(k);
            end
        end

        RadarDataNew(k) = distance;
    end
end
