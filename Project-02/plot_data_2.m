function [Robot, Destiny, Radar ] = plot_data(PositionCurrent, ...
                                              PositionDestiny, ...
                                              Radar, ...
                                              SensorRadius)
    clf
    hold on

    Robot       = plot(PositionCurrent(1), PositionCurrent(2), ...
                       'bo', ...
                       "LineWidth",  2, ...
                       "MarkerSize", 5)

    Destiny     = line([PositionCurrent(1) PositionDestiny(1)], ...
                       [PositionCurrent(2) PositionDestiny(2)], ...
                       'Color',     'green', ...
                       'LineStyle', '--');

    Radar       = draw_radar(PositionCurrent, SensorRadius)

    Detection   = 
end

function Radar = draw_radar(Centre, SensorRadius)
    theta       = linspace(0, 2*pi);

    x           = SensorRadius * cos(theta) + centre(1);
    y           = SensorRadius * sin(theta) + centre(2);

    circle_plot = plot(x, y, 'm');
end

function Detection = draw_detection()
end
