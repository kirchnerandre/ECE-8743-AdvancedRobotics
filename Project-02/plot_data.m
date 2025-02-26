function plot_data(PositionBegin, PositionIntermediate, PositionEnd, ObstaclesData, ObstaclesLength, RadarData, SensorRadius)
    plot(PositionBegin(1), PositionBegin(2), 'r+', "LineWidth", 2, "MarkerSize", 5)
    plot(PositionEnd  (1), PositionEnd  (2), 'r*', "LineWidth", 2, "MarkerSize", 5)

    index_first = 1;

    for i = 1:size(ObstaclesLength, 2)
        index_last = index_first + ObstaclesLength(i) - 1;

        plot(ObstaclesData(1, index_first:index_last), ...
             ObstaclesData(2, index_first:index_last))

        index_first = index_last + 1;
    end

    line([PositionCurrent(1) PositionIntermediate(1)], ...
         [PositionCurrent(2) PositionIntermediate(2)], ...
         'Color',     'green', ...
         'LineStyle', '--');

    line([PositionIntermediate(1) PositionDestiny(1)], ...
         [PositionIntermediate(2) PositionDestiny(2)], ...
         'Color',     'green', ...
         'LineStyle', '--');

    plot(PositionCurrent(1), ...
         PositionCurrent(2), ...
         'bo', ...
         "LineWidth",  2, ...
         "MarkerSize", 5)

    step    = 0.5;
    angles  = (0:step:(360 - step)) .* pi / 180;
    steps   = size(angles, 2);

    x       = SensorRadius * cos(angles) + PositionCurrent(1);
    y       = SensorRadius * sin(angles) + PositionCurrent(2);

    plot_radar_a = plot(x, y, 'm');

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
                plot(RadarData(index_first:index_last) .* cos(angles(index_first:index_last)) + PositionCurrent(1), ...
                     RadarData(index_first:index_last) .* sin(angles(index_first:index_last)) + PositionCurrent(2), ...
                     'Color', 'red')

            index_first = -1;
            index_last  = -1;
        end
    end

    pause(0.01);
end
