function plot_data_2(PositionDestiny, PositionCurrent, RadarData, SensorRadius)
    hold on

    plot(PositionCurrent(1), ...
         PositionCurrent(2), ...
         'bo', ...
         "LineWidth",  2, ...
         "MarkerSize", 5)

    line([PositionCurrent(1) PositionDestiny(1)], ...
         [PositionCurrent(2) PositionDestiny(2)], ...
         'Color',     'green', ...
         'LineStyle', '--');

    step    = 0.5;
    angles  = (0:step:(360 - step)) .* pi / 180;
    steps   = size(angles, 2);

    x       = SensorRadius * cos(angles) + PositionCurrent(1);
    y       = SensorRadius * sin(angles) + PositionCurrent(2);

    plot(x, y, 'm');

    for i = 1:steps
        if RadarData(i) ~= Inf
            plot(RadarData(i) * cos(angles(i)) + PositionCurrent(1), ...
                 RadarData(i) * sin(angles(i)) + PositionCurrent(2), ...
                 'Color', 'red')
        end
    end
end
