function plot_radar_detection(PositionBegin, RadarData, SensorAngle)
    persistent radar_detection

    if ~isempty(radar_detection)
        for i = 1:size(radar_detection, 2)
            delete(radar_detection(i))
        end

        radar_detection = [];
    end

    radar_data  = [ RadarData RadarData(1) ];

    angles      = (1:361) .* pi / 180;
    index_first = -1;
    index_last  = -1;

    for i = 1:360
        if RadarData(i) == Inf
            break
        end
    end

    for j = 0:360
        k = mod(i + j - 1, 360) + 1;

        if RadarData(k) ~= Inf && index_first == -1
            index_first = k;
        end

        if RadarData(k) == Inf && index_first ~= -1 && index_last == -1
            if k == 1
                index_last = 360;
            else
                index_last = k - 1;
            end
        end

        if index_first ~= -1 && index_last ~= -1
            if index_first > index_last
                xs = radar_data(1:index_first) .* cos(angles(1:index_first)) ...
                   + PositionBegin(1);

                ys = radar_data(1:index_first) .* sin(angles(1:index_first)) ...
                   + PositionBegin(2);

                radar_detection_new = plot(xs, ys, 'Color', 'red', "LineWidth", 5);
                radar_detection     = [ radar_detection radar_detection_new ];

                xs = radar_data(index_last:361) .* cos(angles(index_last:361)) ...
                   + PositionBegin(1);

                ys = radar_data(index_last:361) .* sin(angles(index_last:361)) ...
                   + PositionBegin(2);

                radar_detection_new = plot(xs, ys, 'Color', 'red', "LineWidth", 5);
                radar_detection     = [ radar_detection radar_detection_new ];
            else
                xs = radar_data(index_first:index_last) ...
                   .* cos(angles(index_first:index_last)) + PositionBegin(1);

                ys = radar_data(index_first:index_last) ...
                    .* sin(angles(index_first:index_last)) + PositionBegin(2);

                radar_detection_new = plot(xs, ys, 'Color', 'red', "LineWidth", 5);
                radar_detection     = [ radar_detection radar_detection_new ];
            end

            index_first = -1;
            index_last  = -1;
        end
    end
end
