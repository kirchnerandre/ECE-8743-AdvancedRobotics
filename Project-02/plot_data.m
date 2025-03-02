function plot_data(PositionBegin, PositionIntermediate, PositionFinal, ObstaclesData, ObstaclesLength, RadarData, SensorRange, SensorAngle)
    plot_radar_range(PositionBegin, RadarData, SensorRange, SensorAngle)

    plot_radar_detection(PositionBegin, RadarData, SensorAngle)

    line([PositionBegin(1) PositionIntermediate(1)], ...
         [PositionBegin(2) PositionIntermediate(2)], ...
         'Color',     'green', ...
         'LineStyle', '--');

    line([PositionIntermediate(1) PositionFinal(1)], ...
         [PositionIntermediate(2) PositionFinal(2)], ...
         'Color',     'green', ...
         'LineStyle', '--');

    plot(PositionBegin(1), ...
         PositionBegin(2), ...
         'bo', ...
         "LineWidth",  2, ...
         "MarkerSize", 5)
end

function plot_radar_range(PositionBegin, RadarData, SensorRange, SensorAngle)
    persistent radar_range

    if ~isempty(radar_range)
        delete(radar_range)
    end

    angles  = (0:SensorAngle:(360 - SensorAngle)) .* pi / 180;

    x       = SensorRange * cos(angles) + PositionBegin(1);
    y       = SensorRange * sin(angles) + PositionBegin(2);

    radar_range = plot(x, y, 'm');
end

function plot_radar_detection(PositionBegin, RadarData, SensorAngle)
    persistent radar_detection

    if ~isempty(radar_detection)
        for i = 1:size(radar_detection, 2)
            delete(radar_detection(i))
        end
    end

    angles      = (0:SensorAngle:(360 - SensorAngle)) .* pi / 180;
    steps       = size(angles, 2);

    index_first = -1;
    index_last  = -1;

    for i = 1:steps
        if RadarData(i) ~= Inf && index_first == -1
            index_first = i;
        end

        if RadarData(i) == Inf && index_first ~= -1 && index_last == -1
            index_last = i - 1;
        end

        if index_first ~= -1 && index_last ~= -1
            radar_detection_new = plot(RadarData(index_first:index_last) .* cos(angles(index_first:index_last)) + PositionBegin(1), ...
                                       RadarData(index_first:index_last) .* sin(angles(index_first:index_last)) + PositionBegin(2), ...
                                       'Color', 'red', "LineWidth", 5)

            index_first = -1;
            index_last  = -1;

            radar_detection     = [ radar_detection radar_detection_new ];
        end
    end
end
