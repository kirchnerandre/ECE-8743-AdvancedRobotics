function plot_data(PositionBegin, PositionIntermediate, PositionFinal, ObstaclesData, ObstaclesLength, RadarData, SensorRange)
    plot(PositionBegin(1), PositionBegin(2), 'r+', "LineWidth", 2, "MarkerSize", 5)
    plot(PositionFinal(1), PositionFinal(2), 'r*', "LineWidth", 2, "MarkerSize", 5)

    index_first = 1;

    for i = 1:size(ObstaclesLength, 2)
        index_last = index_first + ObstaclesLength(i) - 1;

        plot(ObstaclesData(1, index_first:index_last), ...
             ObstaclesData(2, index_first:index_last))

        index_first = index_last + 1;
    end

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

    step    = 1.0;
    angles  = (0:step:(360 - step)) .* pi / 180;
    steps   = size(angles, 2);

    x       = SensorRange * cos(angles) + PositionBegin(1);
    y       = SensorRange * sin(angles) + PositionBegin(2);

    plot(x, y, 'm');

    index_first = -1;
    index_last  = -1;

    for i = 1:steps
        if RadarData(i) ~= Inf && index_first == -1
            index_first = i;
        end

        if RadarData(i) == Inf && index_last == -1
            index_last = i - 1;
        end

        if index_first ~= -1 && index_last ~= -1
                plot(RadarData(index_first:index_last) .* cos(angles(index_first:index_last)) + PositionBegin(1), ...
                     RadarData(index_first:index_last) .* sin(angles(index_first:index_last)) + PositionBegin(2), ...
                     'Color', 'red', "LineWidth", 5)

            index_first = -1;
            index_last  = -1;
        end
    end
end
