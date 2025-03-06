function plot_radar_range(PositionBegin, SensorRange, SensorAngle)
    persistent radar_range

    if ~isempty(radar_range)
%       delete(radar_range)
    end

    angles  = (0:360) .* pi / 180;

    x       = SensorRange * cos(angles) + PositionBegin(1);
    y       = SensorRange * sin(angles) + PositionBegin(2);

    radar_range = plot(x, y, 'm');
end
